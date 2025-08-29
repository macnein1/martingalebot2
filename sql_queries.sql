-- Martingale Lab SQL Queries
-- Useful queries for analyzing optimization results

-- ============================================
-- BASIC QUERIES
-- ============================================

-- Get best results from latest experiment
SELECT 
    r.score,
    json_extract(r.params_json, '$.overlap_pct') as overlap_pct,
    json_extract(r.params_json, '$.num_orders') as num_orders,
    json_extract(r.payload_json, '$.schedule.volume_pct') as volumes
FROM results r
JOIN experiments e ON r.experiment_id = e.id
WHERE e.id = (SELECT MAX(id) FROM experiments)
ORDER BY r.score ASC
LIMIT 10;

-- Count results by experiment
SELECT 
    e.id,
    e.run_id,
    e.status,
    COUNT(r.id) as result_count,
    MIN(r.score) as best_score,
    AVG(r.score) as avg_score
FROM experiments e
LEFT JOIN results r ON e.id = r.experiment_id
GROUP BY e.id
ORDER BY e.id DESC;

-- ============================================
-- CONSTRAINT ANALYSIS
-- ============================================

-- Find results with constraint violations
SELECT 
    r.id,
    r.score,
    json_extract(r.diagnostics_json, '$.slope_violations') as slope_violations,
    json_extract(r.diagnostics_json, '$.m2_valid') as m2_valid,
    json_extract(r.diagnostics_json, '$.monotone_valid') as monotone_valid
FROM results r
WHERE 
    json_extract(r.diagnostics_json, '$.slope_violations') > 0
    OR json_extract(r.diagnostics_json, '$.m2_valid') = 0
    OR json_extract(r.diagnostics_json, '$.monotone_valid') = 0
ORDER BY r.score ASC
LIMIT 20;

-- Analyze slope violations distribution
SELECT 
    json_extract(diagnostics_json, '$.slope_violations') as violations,
    COUNT(*) as count,
    AVG(score) as avg_score,
    MIN(score) as best_score
FROM results
GROUP BY violations
ORDER BY violations;

-- Check 2dp normalization
SELECT 
    r.id,
    r.score,
    json_extract(r.payload_json, '$.schedule.volume_pct') as volumes,
    CASE 
        WHEN json_extract(r.payload_json, '$.schedule.volume_pct') LIKE '%.___' 
        THEN 'NOT_2DP'
        ELSE '2DP_OK'
    END as normalization_status
FROM results r
WHERE r.experiment_id = (SELECT MAX(id) FROM experiments)
LIMIT 10;

-- ============================================
-- CONFIGURATION ANALYSIS
-- ============================================

-- Best configurations by performance
SELECT 
    cv.config_hash,
    cv.best_score,
    cv.avg_score,
    cv.use_count,
    json_extract(cv.config_json, '$.constraints.slope_cap') as slope_cap,
    json_extract(cv.config_json, '$.constraints.m2_min') as m2_min,
    json_extract(cv.config_json, '$.constraints.m2_max') as m2_max
FROM config_versions cv
WHERE cv.best_score IS NOT NULL
ORDER BY cv.best_score ASC
LIMIT 10;

-- Configuration lineage (parent-child relationships)
SELECT 
    cl.parent_hash,
    cl.child_hash,
    cl.change_type,
    cl.change_description,
    p.best_score as parent_score,
    c.best_score as child_score,
    (p.best_score - c.best_score) as improvement
FROM config_lineage cl
LEFT JOIN config_versions p ON cl.parent_hash = p.config_hash
LEFT JOIN config_versions c ON cl.child_hash = c.config_hash
WHERE p.best_score IS NOT NULL AND c.best_score IS NOT NULL
ORDER BY improvement DESC;

-- Configuration performance over time
SELECT 
    cp.config_hash,
    cp.experiment_id,
    cp.best_score,
    cp.convergence_batch,
    cp.time_to_best,
    cp.created_at
FROM config_performance cp
ORDER BY cp.created_at DESC
LIMIT 20;

-- ============================================
-- PARAMETER ANALYSIS
-- ============================================

-- Overlap vs Score correlation
SELECT 
    json_extract(params_json, '$.overlap_pct') as overlap,
    AVG(score) as avg_score,
    MIN(score) as best_score,
    COUNT(*) as count
FROM results
GROUP BY overlap
ORDER BY overlap;

-- Number of orders vs Score
SELECT 
    json_extract(params_json, '$.num_orders') as orders,
    AVG(score) as avg_score,
    MIN(score) as best_score,
    COUNT(*) as count
FROM results
GROUP BY orders
ORDER BY orders;

-- Penalty weights analysis
SELECT 
    json_extract(params_json, '$.penalty_preset') as preset,
    AVG(score) as avg_score,
    MIN(score) as best_score,
    COUNT(*) as count
FROM results
WHERE json_extract(params_json, '$.penalty_preset') IS NOT NULL
GROUP BY preset;

-- ============================================
-- SCHEDULE ANALYSIS
-- ============================================

-- Analyze volume distributions
WITH volume_analysis AS (
    SELECT 
        r.id,
        r.score,
        json_extract(r.payload_json, '$.schedule.volume_pct[0]') as v0,
        json_extract(r.payload_json, '$.schedule.volume_pct[1]') as v1,
        json_extract(r.payload_json, '$.schedule.q1_share_pct') as q1_share,
        json_extract(r.payload_json, '$.schedule.q4_share_pct') as q4_share
    FROM results r
    WHERE r.score < 100000  -- Only good results
)
SELECT 
    AVG(CAST(v0 as REAL)) as avg_v0,
    AVG(CAST(v1 as REAL)) as avg_v1,
    AVG(CAST(q1_share as REAL)) as avg_q1,
    AVG(CAST(q4_share as REAL)) as avg_q4
FROM volume_analysis;

-- Find schedules with specific characteristics
SELECT 
    r.id,
    r.score,
    json_extract(r.payload_json, '$.schedule.m2') as m2,
    json_extract(r.payload_json, '$.schedule.max_martingale') as max_m,
    json_extract(r.payload_json, '$.schedule.tail_weight') as tail_weight
FROM results r
WHERE 
    CAST(json_extract(r.payload_json, '$.schedule.m2') as REAL) BETWEEN 0.15 AND 0.25
    AND CAST(json_extract(r.payload_json, '$.schedule.tail_weight') as REAL) > 30
ORDER BY r.score ASC
LIMIT 10;

-- ============================================
-- CHECKPOINT & RESUME
-- ============================================

-- Get resumable runs
SELECT 
    run_id,
    batch_idx,
    best_score,
    kept_total,
    created_at
FROM checkpoints
WHERE run_id IN (
    SELECT run_id 
    FROM experiments 
    WHERE status = 'RUNNING'
)
ORDER BY created_at DESC;

-- Checkpoint progress
SELECT 
    run_id,
    MIN(batch_idx) as start_batch,
    MAX(batch_idx) as last_batch,
    MIN(best_score) as best_score,
    MAX(kept_total) as total_candidates
FROM checkpoints
GROUP BY run_id
ORDER BY MAX(created_at) DESC;

-- ============================================
-- PERFORMANCE METRICS
-- ============================================

-- Evaluation speed by experiment
SELECT 
    e.id,
    e.run_id,
    COUNT(r.id) as total_evals,
    (julianday(e.finished_at) - julianday(e.started_at)) * 86400 as duration_sec,
    COUNT(r.id) / ((julianday(e.finished_at) - julianday(e.started_at)) * 86400) as evals_per_sec
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.finished_at IS NOT NULL
GROUP BY e.id
ORDER BY e.id DESC;

-- Memory usage estimation
SELECT 
    COUNT(*) as total_results,
    ROUND(COUNT(*) * 0.1 / 1024.0, 2) as estimated_mb,
    COUNT(DISTINCT experiment_id) as experiments,
    COUNT(DISTINCT json_extract(params_json, '$.seed')) as unique_seeds
FROM results;

-- ============================================
-- CLEANUP QUERIES
-- ============================================

-- Delete old/failed experiments
DELETE FROM results 
WHERE experiment_id IN (
    SELECT id FROM experiments 
    WHERE status = 'FAILED' 
    AND created_at < datetime('now', '-7 days')
);

-- Vacuum database to reclaim space
-- VACUUM;

-- ============================================
-- EXPORT QUERIES
-- ============================================

-- Export best results to CSV format
.mode csv
.output best_results.csv
SELECT 
    score,
    json_extract(params_json, '$.overlap_pct') as overlap_pct,
    json_extract(params_json, '$.num_orders') as num_orders,
    json_extract(payload_json, '$.schedule.volume_pct') as volumes
FROM results
WHERE score < 50000
ORDER BY score ASC;
.output stdout
.mode list