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

from martingale_lab.orchestrator.dca_orchestrator import (
    DCAOrchestrator, DCAConfig, OrchestratorConfig
)
from martingale_lab.storage.experiments_store import ExperimentsStore
from martingale_lab.utils.logging import (
    configure_logging, configure_eval_sampling, get_cli_logger
)

# Use the new centralized logging system
cli_logger = get_cli_logger()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DCA Optimization CLI - İşlemden En Hızlı Çıkış",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    parser.add_argument("--log-every-batch", type=int, default=1,
                       help="Log batch summary every N batches")
    
    # Time constraints
    parser.add_argument("--max-time-sec", type=int, default=None,
                       help="Maximum runtime in seconds (clean shutdown on timeout)")
    
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
        
        # Constraints
        tail_cap=args.tail_cap,
        min_indent_step=args.min_indent_step,
        softmax_temp=args.softmax_temp,
        
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
                "wave_pattern": dca_config.wave_pattern
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
    
    # Use the new batch aggregator for consistent logging
    BatchAggregator.log_batch_summary(
        batch_idx=progress_data['batch'] - 1,  # Convert to 0-based
        total_batches=progress_data['total_batches'],
        best_score=progress_data['best_score'],
        evaluations=progress_data.get('batch_evaluations', 0),
        candidates_kept=progress_data['candidates_kept'],
        prune_mode=progress_data.get('prune_mode', 'unknown'),
        evaluations_per_second=progress_data['evaluations_per_second'],
        log_every_batch=args.log_every_batch
    )


def main() -> int:
    """Main CLI entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
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
        
        # Create configurations
        dca_config, orch_config = create_orchestrator_config(args)
        
        # Create orchestrator
        orchestrator = DCAOrchestrator(
            config=dca_config,
            store=store,
            orch_config=orch_config
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
                "max_time_sec": args.max_time_sec
            }
        )
        
        # Log config summary
        log_config_summary(dca_config, orch_config, run_id)
        
        # Run optimization with timeout handling
        start_time = time.time()
        results = orchestrator.run_optimization(
            progress_callback=lambda data: progress_callback(data, args),
            notes=args.notes or f"CLI optimization run {run_id}",
            max_time_sec=args.max_time_sec
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
                "timeout_reached": stats.get("timeout_reached", False)
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
        if stats.get("timeout_reached", False):
            print(f"⚠️  Stopped due to timeout ({args.max_time_sec}s)", file=sys.stderr)
        print(f"Database: {args.db}", file=sys.stderr)
        
        return 0
        
    except KeyboardInterrupt:
        cli_logger.info("Optimization interrupted by user", extra={"event": "CLI.INTERRUPT"})
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
        return 1


if __name__ == "__main__":
    sys.exit(main())
