"""
Run context module for tracking each optimization run with unique identifiers.
Provides traceability through UUID, seed, and git version tracking.
"""
import uuid
import subprocess
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional


def _git_sha() -> str:
    """Get current git commit short SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunCtx:
    """Run context containing unique identifiers for traceability."""
    run_id: str
    seed: int
    code_version: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'code_version': self.code_version
        }


def make_runctx(seed: Optional[int] = None) -> RunCtx:
    """
    Create a new run context with unique ID, seed, and code version.
    
    Args:
        seed: Optional seed for reproducibility. If None, generates random seed.
        
    Returns:
        RunCtx with unique identifiers
    """
    if seed is None:
        seed = random.randrange(1, 2**31 - 1)
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    return RunCtx(
        run_id=str(uuid.uuid4()),
        seed=seed,
        code_version=_git_sha()
    )
