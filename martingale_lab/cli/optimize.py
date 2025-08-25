#!/usr/bin/env python3
"""
CLI for DCA optimization - "İşlemden En Hızlı Çıkış" system
Command line interface for running martingale optimization experiments.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dataclasses import replace

from martingale_lab.orchestrator.dca_orchestrator import (
    DCAOrchestrator, DCAConfig, OrchestratorConfig
)
from martingale_lab.storage.experiments_store import ExperimentsStore
from martingale_lab.storage.checkpoint_store import CheckpointStore
from martingale_lab.utils.logging import (
    configure_logging, configure_eval_sampling, get_cli_logger
)
from martingale_lab.utils.runctx import make_runctx

# Use the new centralized logging system
cli_logger = get_cli_logger()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DCA Optimization CLI - İşlemden En Hızlı Çıkış",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Orchestrator selection and workers mode
    parser.add_argument("--orchestrator", choices=["dca"], default="dca",
                       help="Select orchestrator implementation (dca)")
    parser.add_argument("--workers-mode", choices=["thread", "process"], default="thread",
                       help="Worker execution mode for evaluation")

    # Search space parameters
    parser.add_argument("--overlap-min", type=float, default=15.0,
                       help="Minimum overlap percentage")
    parser.add_argument("--overlap-max", type=float, default=30.0,
                       help="Maximum overlap percentage")
    parser.add_argument("--orders-min", type=int, default=5,
                       help="Minimum number of orders")
    parser.add_argument("--orders-max", type=int, default=20,
                       help="Maximum number of orders")
    
    # Scoring weights
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha weight (max_need)")
    parser.add_argument("--beta", type=float, default=0.3,
                       help="Beta weight (var_need)")
    parser.add_argument("--gamma", type=float, default=0.2,
                       help="Gamma weight (tail)")
    parser.add_argument("--penalty", type=float, default=0.1,
                       help="Lambda penalty weight")
    
    # Constraints
    parser.add_argument("--tail-cap", type=float, default=0.40,
                       help="Tail cap constraint")
    parser.add_argument("--min-indent-step", type=float, default=0.05,
                       help="Minimum indent step")
    parser.add_argument("--softmax-temp", type=float, default=1.0,
                       help="Softmax temperature")
    parser.add_argument("--first-volume", type=float, default=0.01,
                       help="First order volume pct (fixed).")
    parser.add_argument("--first-indent", type=float, default=0.0,
                       help="First order indent pct (fixed).")
    parser.add_argument("--g-pre-band", type=str, default="1.01,1.20",
                       help="Pre-normalization local growth band as 'lo,hi'.")
    parser.add_argument("--g-post-band", type=str, default="1.01,1.30",
                       help="Post-normalization local growth band as 'lo,hi'.")
    parser.add_argument("--front-cap", type=float, default=5.0,
                       help="Max sum of first K_front volumes in percent.")
    parser.add_argument("--k-front", type=int, default=3,
                       help="K_front (how many first orders are capped).")
    parser.add_argument("--isotonic-tail", choices=["on", "off"], default="off",
                       help="Apply isotonic smoothing on tail (i>=2).")
    
    # Optimization parameters
    parser.add_argument("--batches", type=int, default=100,
                       help="Maximum number of batches")
    parser.add_argument("--batch-size", type=int, default=3000,
                       help="Number of candidates per batch")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--keep-top", type=int, default=10000,
                       help="Keep top K candidates")
    
    # Early stopping and pruning
    parser.add_argument("--disable-early-stop", action="store_true",
                       help="Disable early stopping")
    parser.add_argument("--prune-threshold", type=float, default=2.0,
                       help="Pruning threshold multiplier")
    parser.add_argument("--grace-batches", type=int, default=5,
                       help="Grace period batches before pruning")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    
    # Database and output
    parser.add_argument("--db", type=str, default="db_results/experiments.db",
                       help="Database path")
    parser.add_argument("--notes", type=str, default="",
                       help="Optional experiment notes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Wave pattern
    parser.add_argument("--wave-pattern", action="store_true",
                       help="Enable wave pattern optimization")
    parser.add_argument("--wave-mode", choices=["anchors", "blocks"], default="anchors",
                       help="Volume shape generator: anchors (default) or blocks (wave blocks).")
    parser.add_argument("--anchors", type=int, default=6,
                       help="Number of anchor points for anchors mode (4..8 typical).")
    parser.add_argument("--blocks", type=int, default=3,
                       help="Number of wave blocks for blocks mode.")
    parser.add_argument("--wave-amp-min", type=float, default=0.05,
                       help="Min wave amplitude for blocks mode.")
    parser.add_argument("--wave-amp-max", type=float, default=0.30,
                       help="Max wave amplitude for blocks mode.")

    # Penalty presets/weights
    parser.add_argument("--penalty-preset", choices=["explore", "robust", "tight"], default=None,
                       help="Penalty weight preset; overrides individual weights if set.")
    parser.add_argument("--w-fixed", type=float, default=3.0, help="Weight for first-order fixed penalty")
    parser.add_argument("--w-second", type=float, default=3.0, help="Weight for second<=first penalty")
    parser.add_argument("--w-gband", type=float, default=2.0, help="Weight for g band penalty")
    parser.add_argument("--w-front", type=float, default=3.0, help="Weight for front-load penalty")
    parser.add_argument("--w-tv", type=float, default=1.0, help="Weight for total variation penalty")
    parser.add_argument("--w-wave", type=float, default=1.0, help="Weight for wave penalty")
    
    # Logging configuration
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Console log level")
    parser.add_argument("--log-json", action="store_true",
                       help="Use JSON format for console output")
    parser.add_argument("--log-file", default=None,
                       help="Write detailed logs to file (JSON format)")
    parser.add_argument("--log-eval-sample", type=float, default=0.0,
                       help="Per-evaluation log sampling rate (0.0-1.0)")
    parser.add_argument("--log-every-batch", type=int, default=5,
                       help="Log batch summary every N batches")
    
    # Time constraints
    parser.add_argument("--max-time-sec", type=int, default=None,
                       help="Maximum runtime in seconds (clean shutdown on timeout)")
    
    # Resume functionality
    parser.add_argument("--resume", action="store_true",
                       help="Resume from the latest available checkpoint")
    parser.add_argument("--resume-into", type=str, default=None,
                       help="Resume into a specific run_id (instead of latest)")
    
    return parser.parse_args()


def setup_database(db_path: str) -> ExperimentsStore:
    """Setup database directory and return store."""
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    
    cli_logger.info(
        f"Database directory: {db_dir}",
        extra={"event": "CLI.DB_SETUP", "db_path": db_path}
    )
    
    return ExperimentsStore(db_path)


def create_orchestrator_config(args: argparse.Namespace) -> tuple[DCAConfig, OrchestratorConfig]:
    """Create orchestrator configurations from CLI arguments."""
    
    def _parse_band(s: str):
        try:
            lo_str, hi_str = s.split(",")
            lo, hi = float(lo_str), float(hi_str)
            if not (0.0 < lo < hi):
                raise ValueError
            return lo, hi
        except Exception:
            raise SystemExit(f"--band parse error: expected 'lo,hi' got '{s}'")

    # Parse band strings
    g_pre_lo, g_pre_hi = _parse_band(args.g_pre_band)
    g_post_lo, g_post_hi = _parse_band(args.g_post_band)
    iso_tail = (args.isotonic_tail == "on")

    # Penalty preset (evaluation_engine tarafındaki isimlerle uyumlu)
    penalty_weights = dict(
        w_fixed=args.w_fixed, w_second=args.w_second, w_gband=args.w_gband,
        w_front=args.w_front, w_tv=args.w_tv, w_wave=args.w_wave
    )
    if args.penalty_preset:
        from martingale_lab.optimizer.evaluation_engine import PRESET_WEIGHTS
        if args.penalty_preset not in PRESET_WEIGHTS:
            raise SystemExit(f"unknown preset: {args.penalty_preset}")
        penalty_weights = PRESET_WEIGHTS[args.penalty_preset]

    dca_config = DCAConfig(
        # Search space
        overlap_min=args.overlap_min,
        overlap_max=args.overlap_max,
        orders_min=args.orders_min,
        orders_max=args.orders_max,
        
        # Optimization parameters
        n_candidates_per_batch=args.batch_size,
        max_batches=args.batches,
        top_k_keep=args.keep_top,
        
        # Evaluation weights
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_penalty=args.penalty,
        
        # Wave pattern
        wave_pattern=args.wave_pattern,
        wave_mode=args.wave_mode,
        anchors=args.anchors,
        blocks=args.blocks,
        wave_amp_min=args.wave_amp_min,
        wave_amp_max=args.wave_amp_max,
        
        # Constraints
        tail_cap=args.tail_cap,
        min_indent_step=args.min_indent_step,
        softmax_temp=args.softmax_temp,
        first_volume=args.first_volume,
        first_indent=args.first_indent,
        g_pre_min=g_pre_lo,
        g_pre_max=g_pre_hi,
        g_post_min=g_post_lo,
        g_post_max=g_post_hi,
        front_cap=args.front_cap,
        k_front=args.k_front,
        isotonic_tail=iso_tail,
        
        # Penalty weights
        penalty_preset=args.penalty_preset,
        w_fixed=penalty_weights["w_fixed"],
        w_second=penalty_weights["w_second"],
        w_gband=penalty_weights["w_gband"],
        w_front=penalty_weights["w_front"],
        w_tv=penalty_weights["w_tv"],
        w_wave=penalty_weights["w_wave"],
        
        # Parallelization
        n_workers=args.workers,
        
        # Random seed
        random_seed=args.seed,
        
        # Logging configuration
        log_eval_sample=args.log_eval_sample,
        log_every_batch=args.log_every_batch,
        max_time_sec=args.max_time_sec
    )
    
    orch_config = OrchestratorConfig(
        # Pruning configuration
        prune_enabled=True,
        prune_mode="multiplier",
        prune_multiplier=args.prune_threshold,
        prune_min_keep=50,
        prune_grace_batches=args.grace_batches,
        
        # Early stop configuration
        early_stop_enabled=not args.disable_early_stop,
        early_stop_patience=args.patience,
        early_stop_delta=1e-6,
        
        # Exhaustive mode
        exhaustive_mode=False
    )
    
    return dca_config, orch_config


def log_config_summary(dca_config: DCAConfig, orch_config: OrchestratorConfig, run_id: str) -> None:
    """Log configuration summary."""
    cli_logger.info(
        f"DCA Configuration Summary",
        extra={
            "event": "CLI.CONFIG_SUMMARY",
            "run_id": run_id,
            "search_space": {
                "overlap_range": f"{dca_config.overlap_min}-{dca_config.overlap_max}%",
                "orders_range": f"{dca_config.orders_min}-{dca_config.orders_max}",
            },
            "weights": {
                "alpha": dca_config.alpha,
                "beta": dca_config.beta,
                "gamma": dca_config.gamma,
                "lambda_penalty": dca_config.lambda_penalty
            },
            "optimization": {
                "batch_size": dca_config.n_candidates_per_batch,
                "max_batches": dca_config.max_batches,
                "workers": dca_config.n_workers,
                "wave_pattern": dca_config.wave_pattern,
                "wave_mode": dca_config.wave_mode,
                "anchors": dca_config.anchors,
                "blocks": dca_config.blocks
            },
            "constraints": {
                "first_volume": dca_config.first_volume,
                "first_indent": dca_config.first_indent,
                "g_pre_band": f"{dca_config.g_pre_min},{dca_config.g_pre_max}",
                "g_post_band": f"{dca_config.g_post_min},{dca_config.g_post_max}",
                "front_cap": dca_config.front_cap,
                "k_front": dca_config.k_front,
                "isotonic_tail": dca_config.isotonic_tail
            },
            "penalties": {
                "preset": dca_config.penalty_preset,
                "w_fixed": dca_config.w_fixed,
                "w_second": dca_config.w_second,
                "w_gband": dca_config.w_gband,
                "w_front": dca_config.w_front,
                "w_tv": dca_config.w_tv,
                "w_wave": dca_config.w_wave
            },
            "pruning": {
                "enabled": orch_config.prune_enabled,
                "threshold": orch_config.prune_multiplier,
                "grace_batches": orch_config.prune_grace_batches
            },
            "early_stopping": {
                "enabled": orch_config.early_stop_enabled,
                "patience": orch_config.early_stop_patience
            }
        }
    )


def progress_callback(progress_data: Dict[str, Any], args: argparse.Namespace) -> None:
    """Progress callback for optimization with new batch aggregator."""
    from martingale_lab.utils.logging import BatchAggregator
    
    # Use the new batch aggregator for consistent logging with single-line format
    BatchAggregator.log_batch_summary(
        batch_idx=progress_data['batch'] - 1,  # Convert to 0-based
        total_batches=progress_data['total_batches'],
        best_score=progress_data['best_score'],
        evaluations=progress_data.get('batch_evaluations', 0),
        candidates_kept=progress_data.get('candidates_kept', 0),  # Legacy parameter
        prune_mode=progress_data.get('prune_mode', 'unknown'),
        evaluations_per_second=progress_data['evaluations_per_second'],
        log_every_batch=args.log_every_batch,
        kept_in_this_batch=progress_data.get('kept_in_this_batch'),
        kept_total=progress_data.get('kept_total')
    )


def main() -> int:
    """Main CLI entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Log absolute database path very early
        try:
            db_abs = os.path.abspath(args.db)
            cli_logger.info(
                f"Opening SQLite at: {db_abs}",
                extra={"event": "CLI.DB_OPEN", "db_abs": db_abs}
            )
        except Exception:
            pass
        
        # Configure centralized logging system FIRST
        configure_logging(
            level=args.log_level,
            json_console=args.log_json,
            log_file=args.log_file
        )
        
        # Configure evaluation sampling
        configure_eval_sampling(
            sample_rate=args.log_eval_sample,
            seed=args.seed
        )
        
        # Setup database
        store = setup_database(args.db)
        
        # Create checkpoint store with same database path
        checkpoint_store = CheckpointStore(args.db)
        # Create run context and start run record (for resume listing)
        run_ctx = make_runctx(args.seed)
        try:
            checkpoint_store.start_run(run_ctx, {
                "orchestrator": args.orchestrator,
                "db_path": args.db,
                "notes": args.notes,
            })
        except Exception:
            pass
        
        # Handle resume functionality
        resume_from = None
        if args.resume or args.resume_into:
            # Create a temporary orchestrator to access checkpoint store
            temp_config = DCAConfig()
            temp_orchestrator = DCAOrchestrator(
                config=temp_config, 
                store=store,
                checkpoint_store=checkpoint_store,
                workers_mode=args.workers_mode
            )
            
            if args.resume_into:
                # Resume into specific run_id
                resume_from = args.resume_into
                cli_logger.info(f"Attempting to resume into run_id: {resume_from}")
            else:
                # Find the latest resumable run
                resumable_runs = temp_orchestrator.get_resumable_runs()
                if not resumable_runs:
                    cli_logger.error("No resumable runs found. Use --resume-into <run_id> to specify a specific run.")
                    return 1
                
                # Show available runs
                print("\n=== Available Resumable Runs ===", file=sys.stderr)
                for i, run in enumerate(resumable_runs):
                    print(f"{i+1}. Run ID: {run['run_id']}", file=sys.stderr)
                    print(f"   Started: {run['started_at']}", file=sys.stderr)
                    print(f"   Last Batch: {run['last_batch_idx']}", file=sys.stderr)
                    print(f"   Best Score: {run['best_score']:.6f}" if run['best_score'] else "   Best Score: None", file=sys.stderr)
                    print(f"   Checkpoint: {run['checkpoint_batch']}" if run['checkpoint_batch'] else "   Checkpoint: None", file=sys.stderr)
                    print("", file=sys.stderr)
                
                # Use the most recent run
                resume_from = resumable_runs[0]['run_id']
                cli_logger.info(f"Resuming from most recent run: {resume_from}")
        
        # Create configurations
        dca_config, orch_config = create_orchestrator_config(args)
        
        # Create orchestrator
        orchestrator = DCAOrchestrator(
            config=dca_config,
            store=store,
            run_id=resume_from if resume_from else run_ctx.run_id,  # Use resume run_id or new run_id
            orch_config=orch_config,
            checkpoint_store=checkpoint_store,
            workers_mode=args.workers_mode
        )
        run_id = orchestrator.run_id
        
        # Log startup
        cli_logger.info(
            f"Starting DCA optimization run {run_id}",
            extra={
                "event": "CLI.START",
                "run_id": run_id,
                "db_path": args.db,
                "notes": args.notes,
                "log_level": args.log_level,
                "log_eval_sample": args.log_eval_sample,
                "max_time_sec": args.max_time_sec,
                "resume_from": resume_from
            }
        )
        
        # Log config summary
        log_config_summary(dca_config, orch_config, run_id)
        
        # Run optimization with timeout handling and resume support
        start_time = time.time()
        results = orchestrator.run_optimization(
            progress_callback=lambda data: progress_callback(data, args),
            notes=args.notes or f"CLI optimization run {run_id}",
            max_time_sec=args.max_time_sec,
            resume_from=resume_from
        )
        
        # Log completion
        stats = results["statistics"]
        cli_logger.info(
            f"Optimization completed successfully",
            extra={
                "event": "CLI.COMPLETE",
                "run_id": run_id,
                "experiment_id": results["experiment_id"],
                "best_score": stats["best_score"],
                "total_evaluations": stats["total_evaluations"],
                "total_time": stats["total_time"],
                "evaluations_per_second": stats["evaluations_per_second"],
                "batches_completed": stats["batches_completed"],
                "early_stopped": stats["early_stopped"],
                "candidates_found": stats["candidates_found"],
                "timeout_reached": stats.get("timeout_reached", False),
                "resume_from": resume_from
            }
        )
        
        # Print final summary to stderr for visibility
        print(f"\n=== DCA Optimization Complete ===", file=sys.stderr)
        print(f"Run ID: {run_id}", file=sys.stderr)
        print(f"Experiment ID: {results['experiment_id']}", file=sys.stderr)
        print(f"Best Score: {stats['best_score']:.6f}", file=sys.stderr)
        print(f"Total Evaluations: {stats['total_evaluations']}", file=sys.stderr)
        print(f"Time: {stats['total_time']:.1f}s", file=sys.stderr)
        print(f"Speed: {stats['evaluations_per_second']:.1f} evals/s", file=sys.stderr)
        if resume_from:
            print(f"🔄 Resumed from: {resume_from}", file=sys.stderr)
        if stats.get("timeout_reached", False):
            print(f"⚠️  Stopped due to timeout ({args.max_time_sec}s)", file=sys.stderr)
        print(f"Database: {args.db}", file=sys.stderr)
        try:
            checkpoint_store.finish_run(run_id, status='completed')
        except Exception:
            pass
        
        return 0
        
    except KeyboardInterrupt:
        cli_logger.info("Optimization interrupted by user", extra={"event": "CLI.INTERRUPT"})
        try:
            # Best-effort mark as cancelled
            args = parse_args()
            CheckpointStore(args.db).finish_run(run_id if 'run_id' in locals() else "unknown", status='cancelled')
        except Exception:
            pass
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        cli_logger.error(
            f"Optimization failed: {e}",
            extra={
                "event": "CLI.ERROR",
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        print(f"Error: {e}", file=sys.stderr)
        try:
            args = parse_args()
            CheckpointStore(args.db).finish_run(run_id if 'run_id' in locals() else "unknown", status='failed')
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
