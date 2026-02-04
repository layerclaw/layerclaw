"""Database schema definitions."""

SCHEMA_VERSION = 1

CREATE_TABLES_SQL = """
-- Runs table: Stores metadata for each training run
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    run_name TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    status TEXT DEFAULT 'running',
    config TEXT,
    total_steps INTEGER DEFAULT 0,
    total_checkpoints INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Checkpoints table: Stores checkpoint metadata
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    metrics TEXT,
    has_samples BOOLEAN DEFAULT 0,
    sample_path TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Gradient statistics table: Stores per-layer gradient stats
CREATE TABLE IF NOT EXISTS gradient_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    layer_name TEXT NOT NULL,
    norm REAL,
    mean REAL,
    std REAL,
    max_val REAL,
    min_val REAL,
    num_zeros INTEGER,
    num_nans INTEGER,
    num_infs INTEGER,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE
);

-- System metrics table: Stores CPU/GPU/memory metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    gpu_metrics TEXT,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE
);

-- Anomalies table: Stores detected anomalies
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    anomaly_type TEXT NOT NULL,
    severity TEXT DEFAULT 'medium',
    details TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, step);
CREATE INDEX IF NOT EXISTS idx_gradient_checkpoint ON gradient_stats(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_gradient_layer ON gradient_stats(layer_name);
CREATE INDEX IF NOT EXISTS idx_anomalies_run ON anomalies(run_id, step);
CREATE INDEX IF NOT EXISTS idx_anomalies_type ON anomalies(anomaly_type);
"""
