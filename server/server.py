import json
import asyncio
import logging
import random
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pathlib import Path

from models import (
    RegisterRequest, HeartbeatRequest, IterationCreate,
    AdminBroadcast, AdminAuth, MessageCreate,
    AgentResponse, IterationResponse, new_id, improvement_pct,
)
from names import generate_agent_name, load_used_names
from dedup import fingerprint
import db

logger = logging.getLogger("dflash")

_SEED_PATH = Path(__file__).parent / "seed_train.py"
try:
    SEED_ALGORITHM_CODE = _SEED_PATH.read_text()
except FileNotFoundError:
    logger.warning("seed_train.py not found at %s", _SEED_PATH)
    SEED_ALGORITHM_CODE = ""

_PREPARE_PATH = Path(__file__).parent.parent / "prepare.py"

_config_cache: dict | None = None


async def get_config_cached() -> dict:
    global _config_cache
    if _config_cache is None:
        async with db.connect() as conn:
            _config_cache = await db.get_config(conn)
    return _config_cache


async def get_baseline_score(conn) -> float | None:
    cursor = await conn.execute(
        "SELECT score FROM experiments "
        "WHERE feasible = 1 ORDER BY created_at ASC LIMIT 1"
    )
    row = await cursor.fetchone()
    return row["score"] if row else None


async def verify_admin(req: AdminAuth) -> None:
    config = await get_config_cached()
    if req.admin_key != config.get("admin_key", "dflash-2026"):
        raise HTTPException(status_code=403, detail="Invalid admin key")


async def get_agent_name(conn, agent_id: str) -> str:
    cursor = await conn.execute("SELECT name FROM agents WHERE id = ?", (agent_id,))
    row = await cursor.fetchone()
    return row["name"] if row else "unknown"


# ── WebSocket manager ──

class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, event: dict):
        if not self.connections:
            return
        results = await asyncio.gather(
            *(ws.send_json(event) for ws in self.connections),
            return_exceptions=True,
        )
        self.connections = [
            ws for ws, result in zip(self.connections, results)
            if not isinstance(result, Exception)
        ]


manager = ConnectionManager()


# ── App lifecycle ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    async with db.connect() as conn:
        names = await db.get_all_agent_names(conn)
    load_used_names(names)
    task = asyncio.create_task(periodic_stats())
    yield
    task.cancel()


app = FastAPI(title="dflash Coordination Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


N_STAGNATION = 2
INACTIVE_MINUTES = 20


def inactive_cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=INACTIVE_MINUTES)).isoformat()


# ── Periodic stats ──

async def periodic_stats():
    while True:
        await asyncio.sleep(10)
        try:
            async with db.connect() as conn:
                best = await db.get_global_best(conn)
                baseline = await get_baseline_score(conn)
                cutoff_ts = inactive_cutoff()
                active = await db.get_agent_count(conn, active_only=True, inactive_cutoff=cutoff_ts)
                total_agents = await db.get_agent_count(conn, active_only=False)
                total_exp = (await (await conn.execute("SELECT COUNT(*) as c FROM experiments")).fetchone())["c"]
                total_hyp = (await (await conn.execute("SELECT COUNT(*) as c FROM hypotheses")).fetchone())["c"]

            best_score = best["score"] if best else None
            imp = (improvement_pct(baseline, best_score)
                   if baseline is not None and best_score is not None else 0)

            await manager.broadcast({
                "type": "stats_update",
                "active_agents": active,
                "total_agents": total_agents,
                "total_experiments": total_exp,
                "hypotheses_count": total_hyp,
                "best_score": best_score,
                "baseline_score": baseline,
                "improvement_pct": imp,
                "timestamp": now(),
            })
        except Exception:
            logger.exception("Error in periodic_stats")


# ── Agent endpoints ──

@app.post("/api/agents/register", response_model=AgentResponse)
async def register_agent(req: RegisterRequest):
    agent_id = new_id()
    agent_name = generate_agent_name()
    timestamp = now()

    async with db.connect() as conn:
        await conn.execute(
            "INSERT INTO agents (id, name, registered_at, last_heartbeat, status) VALUES (?, ?, ?, ?, ?)",
            (agent_id, agent_name, timestamp, timestamp, "idle"),
        )
        await conn.commit()

    await manager.broadcast({
        "type": "agent_joined",
        "agent_id": agent_id,
        "agent_name": agent_name,
        "timestamp": timestamp,
    })

    return AgentResponse(
        agent_id=agent_id,
        agent_name=agent_name,
        registered_at=timestamp,
        config={"heartbeat_interval_seconds": 30},
    )


@app.post("/api/agents/{agent_id}/heartbeat")
async def heartbeat(agent_id: str, req: HeartbeatRequest):
    timestamp = now()
    async with db.connect() as conn:
        await conn.execute(
            "UPDATE agents SET last_heartbeat = ?, status = ? WHERE id = ?",
            (timestamp, req.status, agent_id),
        )
        await conn.commit()
    return {"ack": True, "server_time": timestamp}


# ── File serving (agents download prepare.py) ──

@app.get("/api/files/prepare.py")
async def get_prepare_file():
    if _PREPARE_PATH.exists():
        return PlainTextResponse(_PREPARE_PATH.read_text(), media_type="text/x-python")
    raise HTTPException(status_code=404, detail="prepare.py not found")


# ── State endpoint ──

def _pick_inspiration(all_bests, agent_id, active_agent_ids):
    pool = [b for b in all_bests if b["agent_id"] != agent_id and b["agent_id"] in active_agent_ids]
    return random.choice(pool) if pool else None


@app.get("/api/state")
async def get_state(agent_id: str | None = None):
    async with db.connect() as conn:
        global_best = await db.get_global_best(conn)
        baseline = await get_baseline_score(conn)
        cutoff_ts = inactive_cutoff()
        active = await db.get_agent_count(conn, active_only=True, inactive_cutoff=cutoff_ts)
        total_agents = await db.get_agent_count(conn, active_only=False)
        total_exp = (await (await conn.execute("SELECT COUNT(*) as c FROM experiments")).fetchone())["c"]
        total_hyp = (await (await conn.execute("SELECT COUNT(*) as c FROM hypotheses")).fetchone())["c"]

        if agent_id is not None:
            await conn.execute("UPDATE agents SET last_heartbeat = ? WHERE id = ?", (now(), agent_id))
            await conn.commit()

            my_best = await db.get_agent_best(conn, agent_id)
            cursor = await conn.execute(
                "SELECT experiments_completed, runs_since_improvement, improvements FROM agents WHERE id = ?",
                (agent_id,),
            )
            agent_row = await cursor.fetchone()

            my_best_code = my_best["algorithm_code"] if my_best else SEED_ALGORITHM_CODE
            my_best_score = my_best["score"] if my_best else None
            my_best_experiment_id = my_best["experiment_id"] if my_best else None

            if my_best_experiment_id is not None:
                hyp_clause = "AND h.agent_id = ? AND h.target_best_experiment_id = ?"
                hyp_params = [agent_id, my_best_experiment_id]
            else:
                hyp_clause = "AND h.agent_id = ? AND h.target_best_experiment_id IS NULL"
                hyp_params = [agent_id]

            cursor = await conn.execute(
                f"""SELECT h.id, h.title, h.strategy_tag, h.description
                    FROM hypotheses h WHERE 1=1 {hyp_clause}
                    ORDER BY h.created_at DESC LIMIT 20""",
                hyp_params,
            )
            recent_hypotheses = [dict(row) for row in await cursor.fetchall()]

            inspiration_code = None
            inspiration_agent_name = None
            runs_since = agent_row["runs_since_improvement"] if agent_row else 0
            if runs_since >= N_STAGNATION:
                all_bests = await db.list_agent_bests(conn)
                cursor = await conn.execute(
                    "SELECT id FROM agents WHERE last_heartbeat >= ?", (cutoff_ts,)
                )
                active_ids = {row["id"] for row in await cursor.fetchall()}
                chosen = _pick_inspiration(all_bests, agent_id, active_ids)
                if chosen:
                    inspiration_code = chosen["algorithm_code"]
                    inspiration_agent_name = await get_agent_name(conn, chosen["agent_id"])

            leaderboard = await db.compute_leaderboard(conn, inactive_cutoff())

            return {
                "best_score": global_best["score"] if global_best else None,
                "best_algorithm_code": my_best_code,
                "best_experiment_id": my_best_experiment_id,
                "my_best_score": my_best_score,
                "my_runs": agent_row["experiments_completed"] if agent_row else 0,
                "my_improvements": agent_row["improvements"] if agent_row else 0,
                "my_runs_since_improvement": runs_since,
                "active_agents": active,
                "total_agents": total_agents,
                "total_experiments": total_exp,
                "hypotheses_count": total_hyp,
                "recent_hypotheses": [
                    {"id": h["id"], "title": h["title"],
                     "strategy_tag": h["strategy_tag"], "description": h["description"]}
                    for h in recent_hypotheses
                ],
                "inspiration_code": inspiration_code,
                "inspiration_agent_name": inspiration_agent_name,
                "leaderboard": leaderboard,
            }

        # ── Dashboard view ──
        cursor = await conn.execute("""
            SELECT e.*, a.name as agent_name,
                   EXISTS(SELECT 1 FROM best_history bh WHERE bh.experiment_id = e.id) as is_new_best
            FROM experiments e JOIN agents a ON a.id = e.agent_id
            ORDER BY e.created_at DESC LIMIT 20
        """)
        recent_experiments = [dict(row) for row in await cursor.fetchall()]

        cursor = await conn.execute(
            """SELECT h.id, h.title, h.strategy_tag, h.description,
                      a.name as agent_name, h.agent_id
               FROM hypotheses h JOIN agents a ON a.id = h.agent_id
               ORDER BY h.created_at DESC LIMIT 30"""
        )
        recent_hypotheses = [dict(row) for row in await cursor.fetchall()]
        leaderboard = await db.compute_leaderboard(conn, inactive_cutoff())

    global_best_score = global_best["score"] if global_best else None
    overall_imp = (improvement_pct(baseline, global_best_score)
                   if baseline is not None and global_best_score is not None else 0)

    return {
        "baseline_score": baseline,
        "best_score": global_best_score,
        "improvement_pct": overall_imp,
        "best_algorithm_code": global_best["algorithm_code"] if global_best else SEED_ALGORITHM_CODE,
        "active_agents": active,
        "total_agents": total_agents,
        "total_experiments": total_exp,
        "hypotheses_count": total_hyp,
        "recent_experiments": [
            {
                "id": e["id"],
                "agent_name": e["agent_name"],
                "score": e["score"],
                "feasible": bool(e["feasible"]),
                "is_new_best": bool(e["is_new_best"]),
                "beats_own_best": bool(e.get("beats_own_best")),
                "created_at": e["created_at"],
                "notes": e["notes"],
            }
            for e in recent_experiments
        ],
        "recent_hypotheses": [
            {"id": h["id"], "title": h["title"], "strategy_tag": h["strategy_tag"],
             "agent_name": h["agent_name"], "description": h["description"],
             "agent_id": h.get("agent_id", "")}
            for h in recent_hypotheses
        ],
        "leaderboard": leaderboard,
    }


# ── Iteration endpoint ──

@app.post("/api/iterations", response_model=IterationResponse)
async def create_iteration(req: IterationCreate):
    exp_id = new_id()
    hyp_id = new_id()
    timestamp = now()
    fp = fingerprint(req.title, req.strategy_tag)

    async with db.connect() as conn:
        await conn.execute("BEGIN IMMEDIATE")

        prev_best = await db.get_global_best(conn)
        prev_agent_best = await db.get_agent_best(conn, req.agent_id)
        baseline = await get_baseline_score(conn)

        is_new_best = prev_best is None or req.score < prev_best["score"]
        beats_own_best = prev_agent_best is None or req.score < prev_agent_best["score"]

        target_best_experiment_id = prev_agent_best["experiment_id"] if prev_agent_best else None
        hyp_status = "succeeded" if beats_own_best else "failed"

        await conn.execute(
            """INSERT INTO hypotheses
               (id, agent_id, title, description, strategy_tag, status,
                fingerprint, target_best_experiment_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (hyp_id, req.agent_id, req.title, req.description,
             req.strategy_tag, hyp_status, fp, target_best_experiment_id, timestamp),
        )

        delta_vs_best_pct = None
        if prev_best is not None and prev_best["score"] > 0:
            delta_vs_best_pct = round(
                ((prev_best["score"] - req.score) / prev_best["score"]) * 100, 6)
        delta_vs_own_best_pct = None
        if prev_agent_best is not None and prev_agent_best["score"] > 0:
            delta_vs_own_best_pct = round(
                ((prev_agent_best["score"] - req.score) / prev_agent_best["score"]) * 100, 6)

        await conn.execute(
            """INSERT INTO experiments
               (id, agent_id, hypothesis_id, algorithm_code, score, feasible,
                val_loss, num_params, notes,
                delta_vs_best_pct, delta_vs_own_best_pct, beats_own_best, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (exp_id, req.agent_id, hyp_id, req.algorithm_code, req.score,
             1 if req.feasible else 0, req.val_loss, req.num_params,
             req.notes, delta_vs_best_pct, delta_vs_own_best_pct,
             1 if beats_own_best else 0, timestamp),
        )

        if beats_own_best:
            await conn.execute(
                """UPDATE agents SET
                    experiments_completed = experiments_completed + 1,
                    runs_since_improvement = 0,
                    improvements = improvements + 1,
                    best_score = ? WHERE id = ?""",
                (req.score, req.agent_id),
            )
            await db.upsert_agent_best(
                conn, agent_id=req.agent_id, experiment_id=exp_id,
                algorithm_code=req.algorithm_code, score=req.score,
                feasible=req.feasible, val_loss=req.val_loss,
                num_params=req.num_params, updated_at=timestamp,
            )
        else:
            await conn.execute(
                """UPDATE agents SET
                    experiments_completed = experiments_completed + 1,
                    runs_since_improvement = runs_since_improvement + 1
                   WHERE id = ?""",
                (req.agent_id,),
            )

        agent_name = await get_agent_name(conn, req.agent_id)

        if is_new_best:
            await conn.execute(
                """INSERT INTO best_history
                   (experiment_id, agent_id, agent_name, score, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (exp_id, req.agent_id, agent_name, req.score, timestamp),
            )

        await conn.commit()

        cursor = await conn.execute(
            "SELECT experiments_completed, runs_since_improvement, improvements FROM agents WHERE id = ?",
            (req.agent_id,),
        )
        agent_info = dict(await cursor.fetchone())
        leaderboard = await db.compute_leaderboard(conn, inactive_cutoff())
        rank = next((e["rank"] for e in leaderboard if e["agent_id"] == req.agent_id), 0)

    imp = improvement_pct(baseline, req.score) if baseline is not None else 0.0

    await manager.broadcast({
        "type": "experiment_published",
        "experiment_id": exp_id,
        "agent_name": agent_name,
        "agent_id": req.agent_id,
        "score": req.score,
        "feasible": req.feasible,
        "is_new_best": is_new_best,
        "beats_own_best": beats_own_best,
        "strategy_tag": req.strategy_tag,
        "title": req.title,
        "notes": req.notes,
        "timestamp": timestamp,
    })

    if is_new_best:
        await manager.broadcast({
            "type": "new_global_best",
            "experiment_id": exp_id,
            "agent_name": agent_name,
            "agent_id": req.agent_id,
            "score": req.score,
            "improvement_pct": imp,
            "timestamp": timestamp,
        })

    await manager.broadcast({
        "type": "leaderboard_update",
        "entries": leaderboard,
        "timestamp": timestamp,
    })

    return IterationResponse(
        experiment_id=exp_id,
        hypothesis_id=hyp_id,
        is_new_best=is_new_best,
        beats_own_best=beats_own_best,
        rank=rank,
        runs=agent_info["experiments_completed"],
        improvements=agent_info["improvements"],
        runs_since_improvement=agent_info["runs_since_improvement"],
    )


# ── Messages ──

@app.post("/api/messages")
async def create_message(req: MessageCreate):
    msg_id = new_id()
    timestamp = now()
    async with db.connect() as conn:
        await conn.execute(
            "INSERT INTO messages (id, agent_id, agent_name, content, msg_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, req.agent_id, req.agent_name, req.content, req.msg_type, timestamp),
        )
        await conn.commit()

    await manager.broadcast({
        "type": "chat_message",
        "message_id": msg_id,
        "agent_name": req.agent_name,
        "agent_id": req.agent_id,
        "content": req.content,
        "msg_type": req.msg_type,
        "timestamp": timestamp,
    })
    return {"message_id": msg_id, "timestamp": timestamp}


@app.get("/api/messages")
async def list_messages(limit: int = 50):
    async with db.connect() as conn:
        cursor = await conn.execute(
            "SELECT * FROM messages ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in await cursor.fetchall()]


# ── Leaderboard ──

@app.get("/api/leaderboard")
async def get_leaderboard():
    async with db.connect() as conn:
        leaderboard = await db.compute_leaderboard(conn, inactive_cutoff())
    return {"updated_at": now(), "entries": leaderboard}


# ── Replay ──

@app.get("/api/replay")
async def get_replay():
    async with db.connect() as conn:
        cursor = await conn.execute("SELECT * FROM best_history ORDER BY created_at ASC")
        rows = [dict(row) for row in await cursor.fetchall()]
    return [
        {"experiment_id": r["experiment_id"], "agent_id": r.get("agent_id"),
         "agent_name": r["agent_name"], "score": r["score"], "created_at": r["created_at"]}
        for r in rows
    ]


# ── Admin ──

@app.post("/api/admin/broadcast")
async def admin_broadcast(req: AdminBroadcast):
    await verify_admin(req)
    await manager.broadcast({
        "type": "admin_broadcast",
        "message": req.message,
        "priority": req.priority,
        "timestamp": now(),
    })
    return {"sent": True}


# ── WebSocket ──

@app.websocket("/ws/dashboard")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Health ──

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": now()}


# ── Static dashboard ──
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
