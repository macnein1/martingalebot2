"""
Bayesian Optimization for Martingale Strategies

Uses Gaussian Process to intelligently search parameter space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BayesianState:
    """State of Bayesian optimization"""
    X_observed: np.ndarray  # Observed parameter points
    y_observed: np.ndarray  # Observed scores (lower is better)
    best_score: float
    best_params: Dict[str, float]
    iteration: int


class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Process.
    
    More efficient than random search by learning from previous evaluations.
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]] = None,
        bounds: Dict[str, Tuple[float, float]] = None,  # Alias for param_bounds
        n_initial: int = 10,
        acquisition: str = "ei",  # Expected Improvement
        xi: float = 0.01,  # Exploration parameter
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names and (min, max) bounds
            n_initial: Number of random initial points
            acquisition: Acquisition function ("ei", "ucb", "poi")
            xi: Exploration vs exploitation trade-off
            random_state: Random seed
        """
        # Allow both 'bounds' and 'param_bounds' for compatibility
        if param_bounds is None and bounds is not None:
            param_bounds = bounds
        elif param_bounds is None and bounds is None:
            raise ValueError("Either 'param_bounds' or 'bounds' must be provided")
        
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.xi = xi
        
        # Initialize random state
        self.rng = np.random.RandomState(random_state)
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        
        # Best found so far
        self.best_score = float('inf')
        self.best_params = None
        
        # Gaussian Process parameters (simplified)
        self.length_scale = 1.0
        self.noise = 1e-10
        
        self.iteration = 0
    
    def suggest_next(self) -> Dict[str, float]:
        """
        Suggest next point to evaluate.
        
        Returns:
            Dictionary of parameter values to try
        """
        self.iteration += 1
        
        # Initial random exploration
        if len(self.X_observed) < self.n_initial:
            return self._random_sample()
        
        # Use Gaussian Process to suggest next point
        return self._bayesian_suggestion()
    
    def _random_sample(self) -> Dict[str, float]:
        """
        Generate random sample within bounds.
        
        Returns:
            Dictionary of parameter values
        """
        params = {}
        for param_name in self.param_names:
            min_val, max_val = self.param_bounds[param_name]
            params[param_name] = self.rng.uniform(min_val, max_val)
        return params
    
    def _bayesian_suggestion(self) -> Dict[str, float]:
        """
        Use Gaussian Process and acquisition function to suggest next point.
        
        Returns:
            Dictionary of parameter values
        """
        # Convert observations to arrays
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Normalize y for numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) > 0 else 1.0
        y_normalized = (y - y_mean) / y_std
        
        # Generate candidate points
        n_candidates = 1000
        candidates = np.zeros((n_candidates, self.n_params))
        
        for i in range(n_candidates):
            for j, param_name in enumerate(self.param_names):
                min_val, max_val = self.param_bounds[param_name]
                candidates[i, j] = self.rng.uniform(min_val, max_val)
        
        # Compute acquisition values for all candidates
        acquisition_values = []
        for candidate in candidates:
            acq_value = self._compute_acquisition(
                candidate, X, y_normalized, y_mean, y_std
            )
            acquisition_values.append(acq_value)
        
        # Select best candidate
        best_idx = np.argmax(acquisition_values)
        best_candidate = candidates[best_idx]
        
        # Convert to parameter dictionary
        params = {}
        for j, param_name in enumerate(self.param_names):
            params[param_name] = best_candidate[j]
        
        return params
    
    def _compute_acquisition(
        self,
        x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        y_mean: float,
        y_std: float
    ) -> float:
        """
        Compute acquisition function value at point x.
        
        Args:
            x: Point to evaluate
            X: Observed points
            y: Observed values (normalized)
            y_mean: Mean of original y values
            y_std: Std of original y values
            
        Returns:
            Acquisition function value
        """
        # Compute GP mean and variance at x
        mu, sigma = self._gp_predict(x, X, y)
        
        # Convert back from normalized space
        mu_original = mu * y_std + y_mean
        sigma_original = sigma * y_std
        
        if self.acquisition == "ei":
            # Expected Improvement
            return self._expected_improvement(
                mu_original, sigma_original, self.best_score
            )
        elif self.acquisition == "ucb":
            # Upper Confidence Bound (we minimize, so use lower bound)
            return -(mu_original - 2.0 * sigma_original)
        else:  # poi
            # Probability of Improvement
            return self._probability_of_improvement(
                mu_original, sigma_original, self.best_score
            )
    
    def _gp_predict(
        self,
        x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """
        Gaussian Process prediction at point x.
        
        Simplified GP with RBF kernel.
        
        Args:
            x: Point to predict at
            X: Training points
            y: Training values
            
        Returns:
            (mean, std) predictions
        """
        if len(X) == 0:
            return 0.0, 1.0
        
        # Compute kernel between x and X
        k_star = self._rbf_kernel(x.reshape(1, -1), X).flatten()
        
        # Compute kernel matrix for X
        K = self._rbf_kernel(X, X)
        K_inv = self._safe_inverse(K + self.noise * np.eye(len(K)))
        
        # GP mean
        mu = k_star @ K_inv @ y
        
        # GP variance
        k_star_star = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0]
        var = k_star_star - k_star @ K_inv @ k_star
        sigma = np.sqrt(max(0, var))
        
        return float(mu), float(sigma)
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF (Gaussian) kernel.
        
        Args:
            X1: First set of points (n1 x d)
            X2: Second set of points (n2 x d)
            
        Returns:
            Kernel matrix (n1 x n2)
        """
        # Normalize inputs to [0, 1]
        X1_norm = self._normalize_X(X1)
        X2_norm = self._normalize_X(X2)
        
        # Compute pairwise squared distances
        if X1_norm.ndim == 1:
            X1_norm = X1_norm.reshape(1, -1)
        if X2_norm.ndim == 1:
            X2_norm = X2_norm.reshape(1, -1)
        
        sqdist = np.sum(X1_norm**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2_norm**2, axis=1) - \
                 2 * X1_norm @ X2_norm.T
        
        # RBF kernel
        K = np.exp(-0.5 * sqdist / self.length_scale**2)
        return K
    
    def _normalize_X(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range.
        
        Args:
            X: Parameter values
            
        Returns:
            Normalized values
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm = np.zeros_like(X)
        for j, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[param_name]
            if max_val > min_val:
                X_norm[:, j] = (X[:, j] - min_val) / (max_val - min_val)
            else:
                X_norm[:, j] = 0.5
        
        return X_norm
    
    def _safe_inverse(self, K: np.ndarray) -> np.ndarray:
        """
        Safe matrix inverse with regularization.
        
        Args:
            K: Matrix to invert
            
        Returns:
            Inverse matrix
        """
        try:
            return np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add more regularization if singular
            reg = 1e-6
            while reg < 1.0:
                try:
                    return np.linalg.inv(K + reg * np.eye(len(K)))
                except np.linalg.LinAlgError:
                    reg *= 10
            # Fallback to identity
            return np.eye(len(K))
    
    def _expected_improvement(
        self,
        mu: float,
        sigma: float,
        best: float
    ) -> float:
        """
        Expected Improvement acquisition function.
        
        Args:
            mu: GP mean prediction
            sigma: GP std prediction
            best: Best observed value so far
            
        Returns:
            Expected improvement value
        """
        if sigma <= 0:
            return 0.0
        
        # We minimize, so improvement is when mu < best
        improvement = best - mu - self.xi
        Z = improvement / sigma
        
        # Standard normal PDF and CDF
        from scipy.stats import norm
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return float(ei)
    
    def _probability_of_improvement(
        self,
        mu: float,
        sigma: float,
        best: float
    ) -> float:
        """
        Probability of Improvement acquisition function.
        
        Args:
            mu: GP mean prediction
            sigma: GP std prediction
            best: Best observed value so far
            
        Returns:
            Probability of improvement
        """
        if sigma <= 0:
            return 0.0
        
        # We minimize, so improvement is when mu < best
        improvement = best - mu - self.xi
        Z = improvement / sigma
        
        from scipy.stats import norm
        poi = norm.cdf(Z)
        
        return float(poi)
    
    def update(self, params: Dict[str, float], score: float):
        """
        Update optimizer with new observation.
        
        Args:
            params: Parameter values that were evaluated
            score: Resulting score (lower is better)
        """
        # Convert params to array
        x = np.array([params[name] for name in self.param_names])
        self.X_observed.append(x)
        self.y_observed.append(score)
        
        # Update best
        if score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            logger.info(
                f"Bayesian: New best score {score:.2f} at iteration {self.iteration}"
            )
    
    def get_state(self) -> BayesianState:
        """
        Get current optimization state.
        
        Returns:
            BayesianState object
        """
        return BayesianState(
            X_observed=np.array(self.X_observed) if self.X_observed else np.array([]),
            y_observed=np.array(self.y_observed) if self.y_observed else np.array([]),
            best_score=self.best_score,
            best_params=self.best_params,
            iteration=self.iteration
        )
    
    def plot_convergence(self) -> Dict[str, Any]:
        """
        Get convergence data for plotting.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.y_observed:
            return {}
        
        y = np.array(self.y_observed)
        best_so_far = np.minimum.accumulate(y)
        
        return {
            'iterations': list(range(1, len(y) + 1)),
            'scores': y.tolist(),
            'best_scores': best_so_far.tolist(),
            'final_best': float(self.best_score),
            'improvement_rate': float(
                (y[0] - self.best_score) / y[0] * 100 if y[0] > 0 else 0
            )
        }


class BayesianSearchOrchestrator:
    """
    Orchestrator for Bayesian optimization of martingale strategies.
    """
    
    def __init__(
        self,
        evaluation_function: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        n_calls: int = 100,
        n_initial: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize Bayesian search orchestrator.
        
        Args:
            evaluation_function: Function to evaluate strategies
            param_bounds: Parameter bounds
            n_calls: Total number of evaluations
            n_initial: Number of initial random points
            random_state: Random seed
        """
        self.evaluation_function = evaluation_function
        self.param_bounds = param_bounds
        self.n_calls = n_calls
        self.n_initial = n_initial
        
        self.optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            n_initial=n_initial,
            acquisition="ei",
            random_state=random_state
        )
        
        self.history = []
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Returns:
            Best parameters and score
        """
        logger.info(f"Starting Bayesian optimization with {self.n_calls} evaluations")
        
        for i in range(self.n_calls):
            # Get next point to evaluate
            params = self.optimizer.suggest_next()
            
            # Evaluate
            try:
                result = self.evaluation_function(**params)
                score = result['score']
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                score = float('inf')
            
            # Update optimizer
            self.optimizer.update(params, score)
            
            # Store history
            self.history.append({
                'iteration': i + 1,
                'params': params,
                'score': score
            })
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Bayesian progress: {i+1}/{self.n_calls}, "
                    f"best score: {self.optimizer.best_score:.2f}"
                )
        
        # Get final results
        convergence = self.optimizer.plot_convergence()
        
        return {
            'best_params': self.optimizer.best_params,
            'best_score': self.optimizer.best_score,
            'n_evaluations': self.n_calls,
            'convergence': convergence,
            'history': self.history
        }