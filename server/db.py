import aiosqlite
from contextlib import asynccontextmanager
from pathlib import Path
import os

_data_dir = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent)))
DB_PATH = _data_dir / "dflash.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    registered_at TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL,
    status TEXT DEFAULT 'idle',
    experiments_completed INTEGER DEFAULT 0,
    best_score REAL,
    runs_since_improvement INTEGER DEFAULT 0,
    improvements INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    strategy_tag TEXT NOT NULL,
    status TEXT DEFAULT 'failed',
    fingerprint TEXT NOT NULL,
    target_best_experiment_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS agent_bests (
    agent_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    algorithm_code TEXT NOT NULL,
    score REAL NOT NULL,
    feasible INTEGER NOT NULL DEFAULT 1,
    val_loss REAL DEFAULT 0.0,
    num_params REAL DEFAULT 0.0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    hypothesis_id TEXT,
    algorithm_code TEXT DEFAULT '',
    score REAL NOT NULL,
    feasible INTEGER DEFAULT 1,
    val_loss REAL DEFAULT 0.0,
    num_params REAL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    delta_vs_best_pct REAL,
    delta_vs_own_best_pct REAL,
    beats_own_best INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    agent_name TEXT NOT NULL,
    content TEXT NOT NULL,
    msg_type TEXT DEFAULT 'agent',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS best_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    agent_id TEXT,
    agent_name TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""

SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_exp_feasible_score ON experiments(feasible, score);
CREATE INDEX IF NOT EXISTS idx_exp_agent ON experiments(agent_id);
CREATE INDEX IF NOT EXISTS idx_hyp_agent_target ON hypotheses(agent_id, target_best_experiment_id);
CREATE INDEX IF NOT EXISTS idx_agent_bests_score ON agent_bests(feasible, score);
CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at);
"""

DEFAULT_CONFIG = {
    "admin_key": "dflash-2026",
}


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.executescript(SCHEMA_INDEXES)
        for key, value in DEFAULT_CONFIG.items():
            await db.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                (key, value),
            )
        await db.commit()


@asynccontextmanager
async def connect():
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    try:
        yield conn
    finally:
        await conn.close()


async def get_config(conn: aiosqlite.Connection) -> dict:
    cursor = await conn.execute("SELECT key, value FROM config")
    rows = await cursor.fetchall()
    return {row["key"]: row["value"] for row in rows}


async def get_global_best(conn: aiosqlite.Connection) -> dict | None:
    cursor = await conn.execute(
        "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
        "       score, feasible, val_loss, num_params, updated_at "
        "FROM agent_bests WHERE feasible = 1 "
        "ORDER BY score ASC LIMIT 1"
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def get_agent_best(conn: aiosqlite.Connection, agent_id: str) -> dict | None:
    cursor = await conn.execute(
        "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
        "       score, feasible, val_loss, num_params, updated_at "
        "FROM agent_bests WHERE agent_id = ?",
        (agent_id,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def upsert_agent_best(
    conn, agent_id, experiment_id, algorithm_code, score,
    feasible, val_loss, num_params, updated_at,
) -> None:
    await conn.execute(
        """INSERT INTO agent_bests
           (agent_id, experiment_id, algorithm_code, score, feasible,
            val_loss, num_params, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(agent_id) DO UPDATE SET
             experiment_id = excluded.experiment_id,
             algorithm_code = excluded.algorithm_code,
             score = excluded.score,
             feasible = excluded.feasible,
             val_loss = excluded.val_loss,
             num_params = excluded.num_params,
             updated_at = excluded.updated_at""",
        (agent_id, experiment_id, algorithm_code, score,
         1 if feasible else 0, val_loss, num_params, updated_at),
    )


async def list_agent_bests(conn) -> list[dict]:
    cursor = await conn.execute(
        "SELECT agent_id, experiment_id as id, experiment_id, algorithm_code, "
        "       score, feasible, val_loss, num_params, updated_at "
        "FROM agent_bests WHERE feasible = 1 ORDER BY score ASC"
    )
    return [dict(row) for row in await cursor.fetchall()]


async def get_agent_count(conn, active_only=False, inactive_cutoff=None) -> int:
    if active_only:
        cursor = await conn.execute(
            "SELECT COUNT(*) as c FROM agents WHERE last_heartbeat >= ?",
            (inactive_cutoff,),
        )
    else:
        cursor = await conn.execute("SELECT COUNT(*) as c FROM agents")
    return (await cursor.fetchone())["c"]


async def get_all_agent_names(conn) -> set[str]:
    cursor = await conn.execute("SELECT name FROM agents")
    return {row["name"] for row in await cursor.fetchall()}


async def compute_leaderboard(conn, inactive_cutoff=None) -> list[dict]:
    cursor = await conn.execute(
        """SELECT
            a.id as agent_id, a.name as agent_name,
            a.experiments_completed as runs,
            a.improvements as improvements,
            a.runs_since_improvement as runs_since_improvement,
            a.last_heartbeat as last_heartbeat,
            ab.score as best_score
        FROM agents a
        LEFT JOIN agent_bests ab ON ab.agent_id = a.id AND ab.feasible = 1
        ORDER BY best_score IS NULL, best_score ASC, a.name ASC"""
    )
    rows = await cursor.fetchall()
    return [
        {
            "rank": i + 1,
            "agent_id": row["agent_id"],
            "agent_name": row["agent_name"],
            "runs": row["runs"],
            "improvements": row["improvements"],
            "runs_since_improvement": row["runs_since_improvement"],
            "best_score": row["best_score"],
            "active": (row["last_heartbeat"] >= inactive_cutoff
                       if inactive_cutoff and row["last_heartbeat"] else False),
        }
        for i, row in enumerate(rows)
    ]
