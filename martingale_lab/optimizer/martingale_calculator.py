"""
Martingale Calculator Module
Calculates martingale percentages between orders and related metrics
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MartingaleMetrics:
    """Martingale calculation results"""
    order_prices: List[float]
    price_step_percentages: List[float]  # Fiyat düşüş yüzdeleri (diagnostic)
    martingale_percentages: List[float]  # Hacim artış yüzdeleri (gerçek martingale)
    volume_ratios: List[float]
    needpct_per_order: List[float]  # Her emir için NeedPct değerleri
    total_volume: float
    average_martingale: float
    max_martingale: float
    min_martingale: float
    martingale_volatility: float


class MartingaleCalculator:
    """Calculates martingale percentages and related metrics"""
    
    def __init__(self):
        pass
    
    def calculate_martingale_percentages(self, 
                                       base_price: float,
                                       overlap_pct: float,
                                       num_orders: int,
                                       indent_curve: float = 1.0,
                                       martingale_strength: float = 0.5,
                                       volume_skew: str = "late",
                                       fee_pct: float = 0.0,
                                       tp_buffer: float = 0.0,
                                       volume_ratios_override: Optional[List[float]] = None) -> MartingaleMetrics:
        """
        Calculate martingale percentages between orders
        
        Args:
            base_price: Base price for the strategy
            overlap_pct: Total depth percentage (0-100)
            num_orders: Number of orders
            indent_curve: Indent curve factor (0.5-2.0)
            martingale_strength: Martingale strength factor (0-1)
            volume_skew: Volume skew direction ("early" or "late")
            fee_pct: Fee percentage (0-5)
            tp_buffer: Take profit buffer percentage (0-10)
            
        Returns:
            MartingaleMetrics object with calculated values
        """
        
        # Parameter validation
        assert 0 < overlap_pct < 100, f"overlap_pct must be between 0 and 100, got {overlap_pct}"
        assert num_orders >= 2, f"num_orders must be >= 2, got {num_orders}"
        assert 0 <= fee_pct < 5, f"fee_pct must be between 0 and 5, got {fee_pct}"
        assert 0 <= tp_buffer < 10, f"tp_buffer must be between 0 and 10, got {tp_buffer}"
        
        # Calculate order prices with normalized approach
        order_prices, price_step_percentages = self._calculate_order_prices_normalized(
            base_price, overlap_pct, num_orders, indent_curve
        )
        
        # Calculate volume ratios with proper martingale logic (or override)
        if volume_ratios_override is not None and len(volume_ratios_override) == num_orders:
            volume_ratios = list(np.asarray(volume_ratios_override, dtype=float))
            # Derive martingale percent increases from provided volumes
            martingale_percentages = [0.0]
            for k in range(1, num_orders):
                prev = float(volume_ratios[k - 1])
                curr = float(volume_ratios[k])
                inc = (curr / prev - 1.0) * 100.0 if prev > 0 else 0.0
                martingale_percentages.append(inc)
        else:
            volume_ratios, martingale_percentages = self._calculate_volume_schedule(
                num_orders, martingale_strength, volume_skew
            )
        
        # Calculate NeedPct values using real formula
        needpct_per_order = self._calculate_needpct_series(
            order_prices, volume_ratios, fee_pct, tp_buffer
        )
        
        # Calculate statistics for martingale percentages (volume increases)
        if martingale_percentages:
            average_martingale = np.mean(martingale_percentages)
            max_martingale = np.max(martingale_percentages)
            min_martingale = np.min(martingale_percentages)
            martingale_volatility = np.std(martingale_percentages)
        else:
            average_martingale = 0.0
            max_martingale = 0.0
            min_martingale = 0.0
            martingale_volatility = 0.0
        
        return MartingaleMetrics(
            order_prices=order_prices,
            price_step_percentages=price_step_percentages,
            martingale_percentages=martingale_percentages,
            volume_ratios=volume_ratios,
            needpct_per_order=needpct_per_order,
            total_volume=1.0,  # Normalized to 1
            average_martingale=average_martingale,
            max_martingale=max_martingale,
            min_martingale=min_martingale,
            martingale_volatility=martingale_volatility
        )
    
    def _calculate_order_prices_normalized(self, 
                                         base_price: float,
                                         overlap_pct: float,
                                         num_orders: int,
                                         indent_curve: float) -> Tuple[List[float], List[float]]:
        """
        Calculate order prices with normalized approach to guarantee exact overlap_pct
        
        Args:
            base_price: Base price
            overlap_pct: Total depth percentage (0-100)
            num_orders: Number of orders
            indent_curve: Indent curve factor
            
        Returns:
            Tuple of (prices, step_percentages)
        """
        
        # Generate weights for step drops (between orders)
        idx = np.arange(1, num_orders)
        w = np.power(idx, indent_curve) + 1e-9
        
        # Normalize weights to sum to overlap_pct
        step_drops = (overlap_pct * w / w.sum())  # % olarak
        
        # Calculate cumulative drops
        cum_drops = np.concatenate([[0.0], np.cumsum(step_drops)])
        
        # Calculate prices
        prices = base_price * (1.0 - cum_drops/100.0)
        
        return prices.tolist(), step_drops.tolist()
    
    def _calculate_volume_schedule(self,
                                  num_orders: int,
                                  martingale_strength: float = 0.5,
                                  volume_skew: str = "late") -> Tuple[List[float], List[float]]:
        """
        Calculate volume schedule with proper martingale logic
        
        Args:
            num_orders: Number of orders
            martingale_strength: Martingale strength factor (0-1)
            volume_skew: Volume skew direction ("early" or "late")
            
        Returns:
            Tuple of (volume_ratios, martingale_percentages)
        """
        
        if num_orders == 1:
            return [1.0], []
        
        # Generate indices for volume increases (between orders)
        i = np.arange(num_orders - 1)
        
        # Calculate target increase percentages (1% to 100%)
        if volume_skew == "late":
            # Concentrate increases towards later orders
            pct = 0.01 + martingale_strength * 0.99 * ((i + 1) / (num_orders - 1))
        else:
            # Concentrate increases towards earlier orders
            pct = 0.01 + martingale_strength * 0.99 * (1 - (i) / (num_orders - 1))
        
        # Convert to ratios (1.01 to 2.0)
        r = 1.0 + pct
        
        # Build volume sequence
        vols = [1.0]
        for ri in r:
            vols.append(vols[-1] * ri)
        
        # Normalize to sum to 1
        vols = np.array(vols)
        vols /= vols.sum()
        
        return vols.tolist(), (pct * 100).tolist()  # Return percentages
    
    def _calculate_needpct_series(self,
                                 prices: List[float],
                                 volumes: List[float],
                                 fee_pct: float = 0.0,
                                 tp_buffer: float = 0.0) -> List[float]:
        """
        Calculate NeedPct series using real formula
        
        Args:
            prices: Order prices
            volumes: Volume ratios (normalized to sum to 1)
            fee_pct: Fee percentage (0-5)
            tp_buffer: Take profit buffer percentage (0-10)
            
        Returns:
            List of NeedPct values for each order
        """
        
        prices = np.asarray(prices)
        vols = np.asarray(volumes)
        
        s_vol = 0.0  # Cumulative volume
        s_pv = 0.0   # Cumulative price * volume
        need = []
        
        for k in range(len(prices)):
            s_vol += vols[k]
            s_pv += prices[k] * vols[k]
            
            # Calculate weighted average price
            avg = s_pv / s_vol
            
            # Calculate break-even price with fees and buffer
            be = avg * (1 + fee_pct/100 + tp_buffer/100)
            
            # Calculate NeedPct
            need_pct = max(0.0, (be / prices[k] - 1.0) * 100.0)
            need.append(need_pct)
        
        return need
    
    def calculate_needpct_values(self, 
                                martingale_metrics: MartingaleMetrics,
                                fee_pct: float = 0.0,
                                tp_buffer: float = 0.0) -> List[float]:
        """
        Calculate NeedPct values for each order using real formula
        
        Args:
            martingale_metrics: Calculated martingale metrics
            fee_pct: Fee percentage (0-5)
            tp_buffer: Take profit buffer percentage (0-10)
            
        Returns:
            List of NeedPct values for each order
        """
        
        return self._calculate_needpct_series(
            martingale_metrics.order_prices,
            martingale_metrics.volume_ratios,
            fee_pct,
            tp_buffer
        )
    
    def get_strategy_summary(self, 
                           martingale_metrics: MartingaleMetrics) -> Dict[str, Any]:
        """Get comprehensive strategy summary"""
        
        # Calculate NeedPct statistics
        needpct_values = martingale_metrics.needpct_per_order
        if needpct_values:
            needpct_array = np.array(needpct_values)
            needpct_stats = {
                'values': needpct_values,
                'average': np.mean(needpct_array),
                'max': np.max(needpct_array),
                'min': np.min(needpct_array),
                'std': np.std(needpct_array),
                'tail_risk': np.percentile(needpct_array, 95)  # 95th percentile
            }
        else:
            needpct_stats = None
        
        summary = {
            'order_count': len(martingale_metrics.order_prices),
            'price_range': {
                'start': martingale_metrics.order_prices[0],
                'end': martingale_metrics.order_prices[-1],
                'total_drop': ((martingale_metrics.order_prices[0] - martingale_metrics.order_prices[-1]) / martingale_metrics.order_prices[0]) * 100
            },
            'price_step_stats': {
                'values': martingale_metrics.price_step_percentages,
                'average': np.mean(martingale_metrics.price_step_percentages) if martingale_metrics.price_step_percentages else 0,
                'max': np.max(martingale_metrics.price_step_percentages) if martingale_metrics.price_step_percentages else 0,
                'min': np.min(martingale_metrics.price_step_percentages) if martingale_metrics.price_step_percentages else 0
            },
            'martingale_stats': {
                'values': martingale_metrics.martingale_percentages,
                'average': martingale_metrics.average_martingale,
                'max': martingale_metrics.max_martingale,
                'min': martingale_metrics.min_martingale,
                'volatility': martingale_metrics.martingale_volatility
            },
            'volume_stats': {
                'ratios': martingale_metrics.volume_ratios,
                'total_volume': martingale_metrics.total_volume,
                'volume_concentration': 'early' if martingale_metrics.volume_ratios[0] > martingale_metrics.volume_ratios[-1] else 'late'
            }
        }
        
        if needpct_stats:
            summary['needpct_stats'] = needpct_stats
        
        return summary


# Global instance
martingale_calculator = MartingaleCalculator()
