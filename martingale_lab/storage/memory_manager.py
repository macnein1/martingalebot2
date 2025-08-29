"""
Memory Management Utilities
Provides sliding windows and bounded collections to prevent memory leaks.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, TypeVar, Generic
import heapq
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BoundedBestCandidates(Generic[T]):
    """
    Maintains a bounded collection of best candidates with automatic pruning.
    Prevents unbounded memory growth in long-running optimizations.
    """
    
    def __init__(self, max_size: int = 500, score_key: str = "score"):
        """
        Initialize bounded candidate collection.
        
        Args:
            max_size: Maximum number of candidates to keep in memory
            score_key: Key to extract score from candidate dict
        """
        self.max_size = max_size
        self.score_key = score_key
        self._candidates: List[tuple[float, int, Dict[str, Any]]] = []  # Added counter for stable ordering
        self._seen_ids: set = set()
        self.total_added = 0
        self.total_pruned = 0
        self._counter = 0  # For heap stability
    
    def add(self, candidate: Dict[str, Any]) -> bool:
        """
        Add a candidate, automatically pruning if needed.
        
        Args:
            candidate: Candidate dictionary with score
            
        Returns:
            True if candidate was kept, False if pruned
        """
        score = candidate.get(self.score_key, float('inf'))
        stable_id = candidate.get('stable_id')
        
        # Skip duplicates
        if stable_id and stable_id in self._seen_ids:
            return False
        
        self.total_added += 1
        self._counter += 1
        
        # Use a min-heap (negated scores for max-heap behavior)
        # We want to keep the BEST (lowest) scores
        # Include counter for stable comparison when scores are equal
        if len(self._candidates) < self.max_size:
            heapq.heappush(self._candidates, (-score, self._counter, candidate))
            if stable_id:
                self._seen_ids.add(stable_id)
            return True
        else:
            # Check if this candidate is better than the worst we have
            worst_score = -self._candidates[0][0]
            if score < worst_score:
                # Remove worst and add new
                _, _, removed = heapq.heappop(self._candidates)
                removed_id = removed.get('stable_id')
                if removed_id:
                    self._seen_ids.discard(removed_id)
                
                heapq.heappush(self._candidates, (-score, self._counter, candidate))
                if stable_id:
                    self._seen_ids.add(stable_id)
                
                self.total_pruned += 1
                return True
            else:
                self.total_pruned += 1
                return False
    
    def add_batch(self, candidates: List[Dict[str, Any]]) -> int:
        """
        Add multiple candidates at once.
        
        Returns:
            Number of candidates kept
        """
        kept = 0
        for candidate in candidates:
            if self.add(candidate):
                kept += 1
        
        if self.total_pruned > 0 and self.total_added % 1000 == 0:
            logger.debug(
                f"BoundedCandidates: kept={len(self)}/{self.max_size}, "
                f"total_added={self.total_added}, pruned={self.total_pruned}"
            )
        
        return kept
    
    def get_best(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n best candidates sorted by score."""
        # Sort by actual score (not negated)
        sorted_candidates = sorted(self._candidates, key=lambda x: -x[0], reverse=True)
        return [candidate for _, _, candidate in sorted_candidates[:n]]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all candidates sorted by score."""
        sorted_candidates = sorted(self._candidates, key=lambda x: -x[0], reverse=True)
        return [candidate for _, _, candidate in sorted_candidates]
    
    def clear(self):
        """Clear all candidates."""
        self._candidates.clear()
        self._seen_ids.clear()
        logger.debug(f"Cleared {len(self._candidates)} candidates from memory")
    
    def __len__(self) -> int:
        return len(self._candidates)
    
    def __bool__(self) -> bool:
        return bool(self._candidates)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        import sys
        
        # Estimate memory usage
        candidate_size = sum(sys.getsizeof(c[2]) for c in self._candidates)
        seen_ids_size = sys.getsizeof(self._seen_ids)
        
        return {
            "num_candidates": len(self._candidates),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_pruned": self.total_pruned,
            "retention_rate": (1 - self.total_pruned / max(1, self.total_added)) * 100,
            "estimated_memory_mb": (candidate_size + seen_ids_size) / (1024 * 1024),
            "unique_ids": len(self._seen_ids)
        }


class SlidingWindowBuffer(Generic[T]):
    """
    Fixed-size sliding window buffer for streaming data.
    Automatically discards oldest items when full.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize sliding window buffer.
        
        Args:
            max_size: Maximum number of items to keep
        """
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self.total_added = 0
        self.total_discarded = 0
    
    def add(self, item: T):
        """Add item to buffer, discarding oldest if full."""
        if len(self._buffer) >= self.max_size:
            self.total_discarded += 1
        
        self._buffer.append(item)
        self.total_added += 1
    
    def get_recent(self, n: int) -> List[T]:
        """Get n most recent items."""
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]
    
    def get_all(self) -> List[T]:
        """Get all items in buffer."""
        return list(self._buffer)
    
    def clear(self):
        """Clear buffer."""
        self._buffer.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __bool__(self) -> bool:
        return bool(self._buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "current_size": len(self._buffer),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_discarded": self.total_discarded,
            "fill_rate": len(self._buffer) / self.max_size * 100
        }


class BatchAccumulator:
    """
    Accumulates items in batches for efficient processing.
    Prevents memory buildup by forcing batch processing at size limits.
    """
    
    def __init__(self, batch_size: int = 100, max_batches: int = 10):
        """
        Initialize batch accumulator.
        
        Args:
            batch_size: Items per batch
            max_batches: Maximum batches to keep before forcing flush
        """
        self.batch_size = batch_size
        self.max_batches = max_batches
        self._current_batch: List[Any] = []
        self._pending_batches: deque = deque(maxlen=max_batches)
        self.total_items = 0
        self.total_batches = 0
    
    def add(self, item: Any) -> Optional[List[Any]]:
        """
        Add item to accumulator.
        
        Returns:
            Completed batch if ready, None otherwise
        """
        self._current_batch.append(item)
        self.total_items += 1
        
        if len(self._current_batch) >= self.batch_size:
            batch = self._current_batch
            self._current_batch = []
            self.total_batches += 1
            
            # Check if we need to force flush
            if len(self._pending_batches) >= self.max_batches:
                logger.warning(
                    f"BatchAccumulator: Forcing flush at {len(self._pending_batches)} batches"
                )
            
            self._pending_batches.append(batch)
            return batch
        
        return None
    
    def flush(self) -> List[List[Any]]:
        """
        Flush all pending batches including partial current batch.
        
        Returns:
            List of all pending batches
        """
        all_batches = list(self._pending_batches)
        
        if self._current_batch:
            all_batches.append(self._current_batch)
            self._current_batch = []
        
        self._pending_batches.clear()
        
        return all_batches
    
    def get_pending_count(self) -> int:
        """Get number of items waiting to be batched."""
        return len(self._current_batch) + sum(len(b) for b in self._pending_batches)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accumulator statistics."""
        return {
            "current_batch_size": len(self._current_batch),
            "pending_batches": len(self._pending_batches),
            "total_items": self.total_items,
            "total_batches": self.total_batches,
            "pending_items": self.get_pending_count()
        }


def estimate_memory_usage(obj: Any) -> int:
    """
    Estimate memory usage of an object in bytes.
    
    Args:
        obj: Object to measure
        
    Returns:
        Estimated size in bytes
    """
    import sys
    import gc
    
    # Get base size
    size = sys.getsizeof(obj)
    
    # If it's a container, recursively add sizes
    if isinstance(obj, dict):
        size += sum(estimate_memory_usage(k) + estimate_memory_usage(v) 
                   for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(estimate_memory_usage(item) for item in obj)
    
    return size


def log_memory_stats(label: str = ""):
    """Log current memory usage statistics."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    logger.info(
        f"Memory Stats {label}: "
        f"RSS={mem_info.rss / 1024 / 1024:.1f}MB, "
        f"VMS={mem_info.vms / 1024 / 1024:.1f}MB, "
        f"CPU={process.cpu_percent():.1f}%"
    )