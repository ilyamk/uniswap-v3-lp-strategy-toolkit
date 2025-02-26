#!/usr/bin/env python3
"""
Uniswap V3 Liquidity Provision Strategy with Timestamp-Based Rebalancing

This script implements a sophisticated liquidity provision strategy for Uniswap V3 pools
with timestamp-based rebalancing. It uses historical data to determine optimal rebalancing
times and position ranges.

Features:
- Time-based rebalancing (daily, weekly, or custom intervals)
- Volatility-based position sizing
- Multiple rebalancing trigger methods (time, price change, volatility)
- Performance metrics calculation (returns, IL, Sharpe ratio)
- Visualization of strategy performance
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Set style for plots
plt.style.use('fast')
sns.set_palette("viridis")

# Constants
SECONDS_PER_DAY = 86400
SECONDS_PER_WEEK = 604800
DEFAULT_DATA_FILE = "uniswap_v3_data_wstETH_WETH_0_pool_data.csv"
DEFAULT_IL_FILE = "uniswap_v3_data_wstETH_WETH_0.01pct_impermanent_loss.csv"


class LiquidityProvisionStrategy:
    """Class for implementing and backtesting liquidity provision strategies."""
    
    def __init__(self, pool_data_file: str, il_data_file: Optional[str] = None):
        """Initialize with pool data file path."""
        # Load data
        self.pool_data = pd.read_csv(pool_data_file)
        
        # Convert timestamp to datetime
        self.pool_data['datetime'] = pd.to_datetime(self.pool_data['timestamp'], unit='s')
        
        # Sort by timestamp
        self.pool_data = self.pool_data.sort_values('timestamp')
        
        # Load IL data if provided
        self.il_data = None
        if il_data_file and os.path.exists(il_data_file):
            self.il_data = pd.read_csv(il_data_file)
            self.il_data['datetime'] = pd.to_datetime(self.il_data['timestamp'], unit='s')
            self.il_data = self.il_data.sort_values('timestamp')
        
        # Strategy parameters (defaults)
        self.tick_range = 10  # Default tick range (±10 ticks)
        self.rebalance_interval = SECONDS_PER_DAY  # Default: daily rebalancing
        self.price_threshold = 0.5  # Default: 0.5% price change triggers rebalance
        self.volatility_threshold = 1.0  # Default: 1.0 daily volatility triggers rebalance
        self.initial_capital = 100000.0  # Default: $10,000 starting capital
        
        # Strategy results
        self.rebalance_events = []
        self.strategy_returns = []
        self.fees_collected = []
        self.il_incurred = []
        self.portfolio_values = []  # Track portfolio value over time
        self.rebalance_costs = []  # Track costs associated with rebalancing
        
        print(f"Loaded pool data with {len(self.pool_data)} data points")
        if self.il_data is not None:
            print(f"Loaded IL data with {len(self.il_data)} data points")
    
    def set_strategy_params(self, 
                           tick_range: int = 10, 
                           rebalance_interval: int = SECONDS_PER_DAY,
                           price_threshold: float = 0.5,
                           volatility_threshold: float = 1.0,
                           initial_capital: float = 100000.0):
        """Set strategy parameters."""
        self.tick_range = tick_range
        self.rebalance_interval = rebalance_interval
        self.price_threshold = price_threshold
        self.volatility_threshold = volatility_threshold
        self.initial_capital = initial_capital
        
        print(f"Strategy parameters set:")
        print(f"- Initial capital: ${initial_capital:.2f}")
        print(f"- Tick range: ±{tick_range}")
        print(f"- Rebalance interval: {rebalance_interval} seconds")
        print(f"- Price threshold: {price_threshold}%")
        print(f"- Volatility threshold: {volatility_threshold}")
    
    def calculate_optimal_tick_range(self, lookback_days: int = 7) -> int:
        """Calculate optimal tick range based on recent price volatility."""
        # Get recent data
        now = self.pool_data['timestamp'].max()
        lookback_seconds = lookback_days * SECONDS_PER_DAY
        recent_data = self.pool_data[self.pool_data['timestamp'] >= now - lookback_seconds]
        
        if len(recent_data) < 2:
            return self.tick_range  # Not enough data, use default
        
        # Calculate price range
        price_min = recent_data['price'].min()
        price_max = recent_data['price'].max()
        price_mid = (price_min + price_max) / 2
        
        # Calculate tick range needed to cover this price range
        # Price = 1.0001^tick, so tick = log(price) / log(1.0001)
        tick_min = int(np.log(price_min / price_mid) / np.log(1.0001))
        tick_max = int(np.log(price_max / price_mid) / np.log(1.0001))
        
        # Add a safety margin
        tick_range = max(abs(tick_min), abs(tick_max)) + 5
        
        return tick_range
    
    def should_rebalance_time(self, current_timestamp: int, last_rebalance_timestamp: int) -> bool:
        """Check if we should rebalance based on time interval."""
        time_since_last_rebalance = current_timestamp - last_rebalance_timestamp
        return time_since_last_rebalance >= self.rebalance_interval
    
    def should_rebalance_price(self, current_price: float, last_rebalance_price: float) -> bool:
        """Check if we should rebalance based on price change."""
        price_change_pct = abs((current_price - last_rebalance_price) / last_rebalance_price * 100)
        return price_change_pct >= self.price_threshold
    
    def should_rebalance_volatility(self, current_volatility: float) -> bool:
        """Check if we should rebalance based on volatility."""
        return current_volatility >= self.volatility_threshold
    
    def estimate_fees(self, 
                     time_diff_seconds: float, 
                     volatility: float, 
                     liquidity: float, 
                     fee_tier: float = 0.0001) -> float:
        """Estimate fees earned based on time, volatility, and liquidity."""
        # Simple fee estimation model
        # Higher volatility typically means more trading and fees
        time_fraction = time_diff_seconds / SECONDS_PER_DAY  # Fraction of a day
        
        # Base fee rate adjusted by volatility (more volatility = more trading = more fees)
        fee_rate = fee_tier * (1 + volatility / 5)  # Increased volatility impact
        
        # Estimate daily fee as a percentage of liquidity
        # Increase the multiplier to get more realistic fees
        estimated_daily_fee = fee_rate * volatility * float(liquidity) * 0.1  # Increased from 0.01 to 0.1
        
        # Add a minimum fee based on liquidity to ensure some fees are always collected
        min_daily_fee = float(liquidity) * fee_tier * 0.05
        estimated_daily_fee = max(estimated_daily_fee, min_daily_fee)
        
        # Scale by time fraction
        fee_collected = estimated_daily_fee * time_fraction
        
        return fee_collected
    
    def calculate_realistic_fees(self,
                                position_liquidity: float,
                                pool_liquidity: float,
                                time_diff_seconds: float,
                                tick_lower: int,
                                tick_upper: int,
                                current_tick: int,
                                fee_tier: float,
                                daily_volume: Optional[float] = None,
                                tick_data: Optional[Dict[int, float]] = None) -> float:
        """
        Calculate fees in a more realistic way using position size relative to pool liquidity
        and optional trading volume data.
        
        Args:
            position_liquidity: Liquidity of the position
            pool_liquidity: Total pool liquidity
            time_diff_seconds: Time difference in seconds
            tick_lower: Lower tick of position
            tick_upper: Upper tick of position
            current_tick: Current tick of pool
            fee_tier: Fee tier of pool (e.g., 0.0001 for 0.01% pool)
            daily_volume: Optional daily trading volume
            tick_data: Optional dictionary mapping ticks to liquidity amounts
            
        Returns:
            Estimated fee earnings for the period
        """
        # Check if position is in range
        in_range = tick_lower <= current_tick <= tick_upper
        if not in_range:
            return 0.0  # No fees if position is out of range
            
        # Calculate proportion of position relative to total pool liquidity
        if pool_liquidity <= 0:
            return 0.0
            
        liquidity_proportion = position_liquidity / pool_liquidity
        
        # If we have tick data, calculate more precise proportion based on active ticks
        if tick_data is not None:
            active_liquidity = sum([
                liq for tick, liq in tick_data.items() 
                if tick_lower <= tick <= tick_upper
            ])
            if active_liquidity > 0:
                liquidity_proportion = position_liquidity / active_liquidity
        
        # Calculate fee earnings based on volume if available
        if daily_volume is not None:
            # Estimate volume for time period
            period_volume = daily_volume * (time_diff_seconds / SECONDS_PER_DAY)
            
            # Calculate fees
            # Pool fee is applied to volume, liquidity providers get a portion
            # LP proportion is typically 60-70% of total fees
            lp_fee_share = 0.7  # 70% of fees go to LPs
            period_fee = period_volume * fee_tier * lp_fee_share
            
            # Position's share of period fee
            position_fee = period_fee * liquidity_proportion
            
            return position_fee
        else:
            # Fall back to volatility-based estimation
            # This is just a fallback estimation
            time_fraction = time_diff_seconds / SECONDS_PER_DAY
            estimated_volatility = 1.0  # Default volatility
            
            # Try to get volatility from current data point
            if hasattr(self, 'current_data_point') and 'est_volatility_daily' in self.current_data_point:
                estimated_volatility = self.current_data_point['est_volatility_daily']
            
            # Estimate daily fee as percentage of liquidity
            fee_rate = fee_tier * (1 + estimated_volatility / 10)
            daily_fee = fee_rate * estimated_volatility * pool_liquidity * 0.01
            
            # Apply liquidity proportion and time fraction
            position_fee = daily_fee * liquidity_proportion * time_fraction
            
            return position_fee
    
    def calculate_swap_slippage(self, 
                               amount: float, 
                               liquidity: float, 
                               price_impact_factor: float = 0.05) -> float:
        """
        Calculate slippage for swapping tokens during rebalancing.
        
        Args:
            amount: Amount being swapped in USD
            liquidity: Current pool liquidity
            price_impact_factor: Factor to adjust price impact calculation
            
        Returns:
            Slippage cost in USD
        """
        # Calculate slippage based on amount relative to pool liquidity
        # Higher amounts relative to liquidity cause more slippage
        if liquidity == 0:
            return 0
            
        # Convert liquidity to approximate USD value
        liquidity_usd = float(liquidity) * 0.01  # Simple conversion factor
        
        # Calculate price impact
        price_impact = price_impact_factor * (amount / liquidity_usd) ** 2
        
        # Cap price impact at reasonable levels
        price_impact = min(price_impact, 0.1)  # Max 10% impact
        
        # Calculate cost
        slippage_cost = amount * price_impact
        
        return slippage_cost
    
    def calculate_rebalance_cost(self, 
                                current_price: float,
                                new_tick_low: int,
                                new_tick_high: int,
                                position_value: float,
                                gas_price_gwei: float = 30.0,
                                gas_used: int = 250000) -> Dict[str, float]:
        """
        Calculate the cost of rebalancing a liquidity position.
        
        Args:
            current_price: Current token price
            new_tick_low: New lower tick bound
            new_tick_high: New upper tick bound
            position_value: Value of position in USD
            gas_price_gwei: Gas price in Gwei
            gas_used: Gas used for rebalancing transaction
            
        Returns:
            Dictionary containing swap_fee, slippage, gas_cost and total_cost
        """
        # Calculate transaction fee on DEX (usually 0.05% for swaps)
        swap_fee = position_value * 0.0005
        
        # Calculate slippage
        slippage = self.calculate_swap_slippage(position_value, self.pool_data.iloc[-1]['liquidity'])
        
        # Calculate gas cost in ETH
        gas_cost_eth = (gas_price_gwei * 1e-9) * gas_used
        
        # Convert gas cost to USD using current price
        # Assuming ETH is token1 (usually the case for ETH pairs)
        gas_cost_usd = gas_cost_eth * current_price
        
        # Total cost
        total_cost = swap_fee + slippage + gas_cost_usd
        
        return {
            "swap_fee": swap_fee,
            "slippage": slippage,
            "gas_cost": gas_cost_usd,
            "total_cost": total_cost
        }
    
    def calculate_il(self, start_price: float, end_price: float) -> float:
        """Calculate impermanent loss for a price change."""
        price_ratio = end_price / start_price
        il_pct = (2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1) * 100
        return il_pct
    
    def calculate_drawdown(self, portfolio_values: List[float]) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            portfolio_values: List of portfolio values over time
            
        Returns:
            Dictionary with maximum drawdown, drawdown duration, and recovery time
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return {"max_drawdown_pct": 0, "max_drawdown_duration": 0, "recovery_time": 0}
            
        # Calculate drawdown series
        peak = portfolio_values[0]
        drawdowns = []
        drawdown_durations = []
        current_drawdown = 0
        current_duration = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown = 0
                current_duration = 0
            else:
                # Calculate percentage drawdown
                current_drawdown = (peak - value) / peak * 100
                current_duration += 1
                
            drawdowns.append(current_drawdown)
            drawdown_durations.append(current_duration)
            
        # Find maximum drawdown and its duration
        if not drawdowns:
            return {"max_drawdown_pct": 0, "max_drawdown_duration": 0, "recovery_time": 0}
            
        max_drawdown = max(drawdowns)
        max_idx = drawdowns.index(max_drawdown)
        max_duration = drawdown_durations[max_idx]
        
        # Calculate recovery time (time from max drawdown to new peak)
        recovery_time = 0
        if max_idx < len(portfolio_values) - 1:
            peak_after_drawdown = peak
            for i in range(max_idx + 1, len(portfolio_values)):
                recovery_time += 1
                if portfolio_values[i] >= peak_after_drawdown:
                    break
        
        return {
            "max_drawdown_pct": max_drawdown,
            "max_drawdown_duration": max_duration,
            "recovery_time": recovery_time
        }
    
    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio, which only penalizes downside risk.
        
        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate for the period
            
        Returns:
            Sortino ratio
        """
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Calculate downside deviation (standard deviation of negative returns only)
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return float('inf')  # No negative returns
            
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return float('inf')  # Avoid division by zero
            
        sortino = (mean_return - risk_free_rate) / downside_deviation
        return sortino
    
    def calculate_calmar_ratio(self, annualized_return: float, max_drawdown_pct: float) -> float:
        """
        Calculate Calmar ratio (annualized return divided by maximum drawdown).
        
        Args:
            annualized_return: Annualized return percentage
            max_drawdown_pct: Maximum drawdown percentage
            
        Returns:
            Calmar ratio
        """
        if max_drawdown_pct == 0:
            return float('inf')  # No drawdown
            
        calmar = annualized_return / max_drawdown_pct
        return calmar
    
    def calculate_win_loss_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate win/loss metrics like win rate, profit factor, and average win/loss.
        
        Args:
            returns: List of period returns
            
        Returns:
            Dictionary with win/loss metrics
        """
        if not returns or len(returns) < 2:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0
            }
            
        returns_array = np.array(returns)
        
        # Separate winning and losing trades
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        total_count = len(returns_array)
        
        # Calculate win rate
        win_rate = win_count / total_count if total_count > 0 else 0
        
        # Calculate profit factor (gross profits / gross losses)
        gross_profits = np.sum(wins) if win_count > 0 else 0
        gross_losses = abs(np.sum(losses)) if loss_count > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Calculate average win and loss
        avg_win = np.mean(wins) if win_count > 0 else 0
        avg_loss = abs(np.mean(losses)) if loss_count > 0 else 0
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio
        }
    
    def backtest_time_based_strategy(self) -> Dict[str, Any]:
        """
        Backtest a time-based liquidity provision strategy.
        
        Strategy:
        1. Provide liquidity in a range around the current price (±tick_range)
        2. Rebalance at fixed time intervals
        """
        # Reset results
        self.rebalance_events = []
        self.strategy_returns = []
        self.fees_collected = []
        self.il_incurred = []
        self.portfolio_values = []  # Add portfolio values tracking
        
        # Sort data by timestamp
        df = self.pool_data.sort_values('timestamp')
        
        # Initialize strategy variables
        initial_price = df.iloc[0]['price']
        current_tick_range = self.calculate_optimal_tick_range()
        current_range_low = df.iloc[0]['tick'] - current_tick_range
        current_range_high = df.iloc[0]['tick'] + current_tick_range
        last_rebalance_price = initial_price
        last_rebalance_timestamp = df.iloc[0]['timestamp']
        last_rebalance_index = 0
        
        # Initialize portfolio
        current_capital = self.initial_capital
        position_liquidity = current_capital * 0.01  # Simple conversion factor, similar to adaptive strategy
        self.portfolio_values.append(current_capital)
        
        # Track positions and rebalancing
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            current_price = current['price']
            current_timestamp = current['timestamp']
            
            # Check if we need to rebalance based on time
            if self.should_rebalance_time(current_timestamp, last_rebalance_timestamp):
                # Calculate new optimal tick range
                current_tick_range = self.calculate_optimal_tick_range()
                
                # Record rebalance event
                self.rebalance_events.append({
                    "timestamp": current_timestamp,
                    "datetime": current['datetime'],
                    "old_range_low": current_range_low,
                    "old_range_high": current_range_high,
                    "new_range_low": current['tick'] - current_tick_range,
                    "new_range_high": current['tick'] + current_tick_range,
                    "price": current_price,
                    "trigger": "time",
                    "time_since_last": current_timestamp - last_rebalance_timestamp
                })
                
                # Update range and last rebalance info
                current_range_low = current['tick'] - current_tick_range
                current_range_high = current['tick'] + current_tick_range
                
                # Calculate fees and IL for the period
                if 'est_volatility_daily' in current and 'time_diff_seconds' in current:
                    # Calculate time since last rebalance
                    time_diff = current_timestamp - last_rebalance_timestamp
                    
                    # Estimate fees - use position_liquidity instead of pool liquidity
                    volatility = current.get('est_volatility_daily', 0.5)  # Default if missing
                    fee_collected = self.estimate_fees(
                        time_diff, 
                        volatility, 
                        position_liquidity,  # Use position liquidity instead of pool liquidity
                        fee_tier=0.0001  # 0.01% pool
                    )
                    self.fees_collected.append(fee_collected)
                    
                    # Add fees to capital
                    current_capital += fee_collected
                    
                    # Calculate IL
                    il = self.calculate_il(last_rebalance_price, current_price)
                    il_usd = (current_capital * il / 100)  # Convert percentage to USD
                    self.il_incurred.append(il_usd)
                    
                    # Apply IL to capital
                    current_capital -= abs(il_usd)
                    
                    # Calculate net return
                    net_return = fee_collected - abs(il_usd)
                    self.strategy_returns.append(net_return)
                
                # Update position liquidity based on new capital
                position_liquidity = current_capital * 0.01
                
                # Update last rebalance info
                last_rebalance_price = current_price
                last_rebalance_timestamp = current_timestamp
                last_rebalance_index = i
            else:
                # Even if we don't rebalance, we still earn fees and experience IL
                if 'time_diff_seconds' in current:
                    time_diff = current['time_diff_seconds']
                    
                    # Calculate fees - use position_liquidity instead of pool liquidity
                    volatility = current.get('est_volatility_daily', 0.5)  # Default if missing
                    fee_collected = self.estimate_fees(
                        time_diff, 
                        volatility, 
                        position_liquidity,  # Use position liquidity instead of pool liquidity
                        fee_tier=0.0001  # 0.01% pool
                    )
                    self.fees_collected.append(fee_collected)
                    
                    # Add fees to capital
                    current_capital += fee_collected
                    
                    # Calculate IL from previous data point
                    if previous is not None:
                        il = self.calculate_il(previous['price'], current_price)
                        il_usd = (current_capital * il / 100)  # Convert percentage to USD
                        self.il_incurred.append(il_usd)
                        
                        # Apply IL to capital
                        current_capital -= abs(il_usd)
                        
                        # Calculate net return
                        net_return = fee_collected - abs(il_usd)
                        self.strategy_returns.append(net_return)
                
                # Update position liquidity based on new capital
                position_liquidity = current_capital * 0.01
            
            # Update portfolio value tracking
            self.portfolio_values.append(current_capital)
        
        # Calculate strategy metrics
        total_fees = sum(self.fees_collected)
        total_il = sum(self.il_incurred)
        net_return = total_fees - abs(total_il)
        
        # Calculate return metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1] if self.portfolio_values else initial_value
        total_return_pct = ((final_value / initial_value) - 1) * 100
        
        # Calculate annualized metrics
        days_elapsed = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']) / SECONDS_PER_DAY
        annualized_return = net_return * (365 / days_elapsed) if days_elapsed > 0 else 0
        
        # Calculate Sharpe ratio
        if self.strategy_returns and np.std(self.strategy_returns) > 0:
            sharpe = np.mean(self.strategy_returns) / np.std(self.strategy_returns) * np.sqrt(365)
        else:
            sharpe = 0
        
        # Calculate additional performance metrics if we have portfolio values
        drawdown_metrics = self.calculate_drawdown(self.portfolio_values) if self.portfolio_values else {"max_drawdown_pct": 0, "max_drawdown_duration": 0, "recovery_time": 0}
        
        # Calculate win/loss metrics
        win_loss_metrics = self.calculate_win_loss_metrics(self.strategy_returns) if self.strategy_returns else {"win_rate": 0, "profit_factor": 0, "win_loss_ratio": 0}
        
        return {
            "rebalance_events": self.rebalance_events,
            "rebalance_count": len(self.rebalance_events),
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "total_return_pct": total_return_pct,
            "total_fees_collected": total_fees,
            "total_il_incurred": total_il,
            "net_return": net_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": 0,  # Not calculated in this strategy
            "calmar_ratio": 0,   # Not calculated in this strategy
            "max_drawdown_pct": drawdown_metrics["max_drawdown_pct"],
            "win_rate": win_loss_metrics["win_rate"],
            "profit_factor": win_loss_metrics["profit_factor"],
            "win_loss_ratio": win_loss_metrics["win_loss_ratio"],
            "avg_time_between_rebalances": np.mean([e["time_since_last"] for e in self.rebalance_events[1:]]) / SECONDS_PER_DAY if len(self.rebalance_events) > 1 else 0
        }
    
    def backtest_adaptive_strategy(self, 
                             daily_volume: Optional[float] = None, 
                             tick_data: Optional[Dict[str, Dict[int, float]]] = None) -> Dict[str, Any]:
        """
        Backtest an adaptive liquidity provision strategy.
        
        Strategy:
        1. Provide liquidity in a range around the current price (±tick_range)
        2. Rebalance based on multiple triggers:
           - Time interval
           - Price change threshold
           - Volatility threshold
        3. Adjust position range based on recent volatility
        
        Args:
            daily_volume: Optional average daily trading volume (USD)
            tick_data: Optional dict mapping block numbers to tick liquidity data
            
        Returns:
            Dictionary with strategy results
        """
        # Reset results
        self.rebalance_events = []
        self.strategy_returns = []
        self.fees_collected = []
        self.il_incurred = []
        self.portfolio_values = []
        self.rebalance_costs = []
        
        # Sort data by timestamp
        df = self.pool_data.sort_values('timestamp')
        
        # Initialize strategy variables
        initial_price = df.iloc[0]['price']
        current_tick_range = self.calculate_optimal_tick_range()
        current_range_low = df.iloc[0]['tick'] - current_tick_range
        current_range_high = df.iloc[0]['tick'] + current_tick_range
        last_rebalance_price = initial_price
        last_rebalance_timestamp = df.iloc[0]['timestamp']
        last_rebalance_index = 0
        
        # Initialize portfolio
        current_capital = self.initial_capital
        position_liquidity = current_capital * 0.01  # Simple conversion factor, similar to adaptive strategy
        self.portfolio_values.append(current_capital)
        
        # Track block numbers for tick data lookup
        current_block = None
        
        # Track positions and rebalancing
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Store current data point for use in other methods
            self.current_data_point = current
            
            current_price = current['price']
            current_timestamp = current['timestamp']
            current_volatility = current.get('est_volatility_daily', 0)
            
            # Get current block number for tick data lookup
            if 'block' in current:
                current_block = current['block']
            
            # Check rebalance triggers
            time_trigger = self.should_rebalance_time(current_timestamp, last_rebalance_timestamp)
            price_trigger = self.should_rebalance_price(current_price, last_rebalance_price)
            volatility_trigger = self.should_rebalance_volatility(current_volatility)
            
            # Determine if we should rebalance
            should_rebalance = time_trigger or price_trigger or volatility_trigger
            trigger_reason = "none"
            
            if time_trigger:
                trigger_reason = "time"
            elif price_trigger:
                trigger_reason = "price"
            elif volatility_trigger:
                trigger_reason = "volatility"
            
            # Get current tick specific data if available
            current_tick_data = None
            if tick_data is not None and current_block is not None and str(current_block) in tick_data:
                current_tick_data = tick_data[str(current_block)]
            
            if should_rebalance:
                # Calculate new optimal tick range based on recent volatility
                current_tick_range = self.calculate_optimal_tick_range()
                
                # Calculate rebalancing costs
                rebalance_cost = self.calculate_rebalance_cost(
                    current_price=current_price,
                    new_tick_low=current['tick'] - current_tick_range,
                    new_tick_high=current['tick'] + current_tick_range,
                    position_value=current_capital
                )
                
                # Deduct rebalancing costs from capital
                current_capital -= rebalance_cost["total_cost"]
                self.rebalance_costs.append(rebalance_cost)
                
                # Record rebalance event
                self.rebalance_events.append({
                    "timestamp": current_timestamp,
                    "datetime": current['datetime'],
                    "old_range_low": current_range_low,
                    "old_range_high": current_range_high,
                    "new_range_low": current['tick'] - current_tick_range,
                    "new_range_high": current['tick'] + current_tick_range,
                    "price": current_price,
                    "trigger": trigger_reason,
                    "time_since_last": current_timestamp - last_rebalance_timestamp,
                    "price_change_pct": ((current_price - last_rebalance_price) / last_rebalance_price) * 100,
                    "volatility": current_volatility,
                    "rebalance_cost": rebalance_cost["total_cost"],
                    "swap_fee": rebalance_cost["swap_fee"],
                    "slippage": rebalance_cost["slippage"],
                    "gas_cost": rebalance_cost["gas_cost"],
                    "portfolio_value": current_capital
                })
                
                # Update range and last rebalance info
                current_range_low = current['tick'] - current_tick_range
                current_range_high = current['tick'] + current_tick_range
                
                # Calculate fees and IL for the period
                if 'time_diff_seconds' in current:
                    # Calculate time since last rebalance
                    time_diff = current_timestamp - last_rebalance_timestamp
                    
                    # Try to use realistic fee calculation if we have needed data
                    if daily_volume is not None or current_tick_data is not None:
                        fee_collected = self.calculate_realistic_fees(
                            position_liquidity=position_liquidity,
                            pool_liquidity=float(current['liquidity']),
                            time_diff_seconds=time_diff,
                            tick_lower=current_range_low,
                            tick_upper=current_range_high,
                            current_tick=current['tick'],
                            fee_tier=0.0001,  # 0.01% pool
                            daily_volume=daily_volume,
                            tick_data=current_tick_data
                        )
                    else:
                        # Fall back to simpler estimation
                        fee_collected = self.estimate_fees(
                            time_diff, 
                            current_volatility, 
                            position_liquidity,
                            fee_tier=0.0001  # 0.01% pool
                        )
                    
                    self.fees_collected.append(fee_collected)
                    
                    # Add fees to capital
                    current_capital += fee_collected
                    
                    # Calculate IL
                    il = self.calculate_il(last_rebalance_price, current_price)
                    il_usd = (current_capital * il / 100)
                    self.il_incurred.append(il_usd)
                    
                    # Apply IL to capital
                    current_capital -= abs(il_usd)
                    
                    # Calculate net return
                    net_return = fee_collected - abs(il_usd)
                    self.strategy_returns.append(net_return)
                
                # Update position liquidity based on new capital
                position_liquidity = current_capital * 0.01
                
                # Update last rebalance info
                last_rebalance_price = current_price
                last_rebalance_timestamp = current_timestamp
                last_rebalance_index = i
            else:
                # Even if we don't rebalance, we still earn fees and experience IL
                if 'time_diff_seconds' in current:
                    time_diff = current['time_diff_seconds']
                    
                    # Calculate fees - use position_liquidity instead of pool liquidity
                    volatility = current.get('est_volatility_daily', 0.5)  # Default if missing
                    fee_collected = self.estimate_fees(
                        time_diff, 
                        volatility, 
                        position_liquidity,  # Use position liquidity instead of pool liquidity
                        fee_tier=0.0001  # 0.01% pool
                    )
                    self.fees_collected.append(fee_collected)
                    
                    # Add fees to capital
                    current_capital += fee_collected
                    
                    # Calculate IL from previous data point
                    if previous is not None:
                        il = self.calculate_il(previous['price'], current_price)
                        il_usd = (current_capital * il / 100)  # Convert percentage to USD
                        self.il_incurred.append(il_usd)
                        
                        # Apply IL to capital
                        current_capital -= abs(il_usd)
                        
                        # Calculate net return
                        net_return = fee_collected - abs(il_usd)
                        self.strategy_returns.append(net_return)
                
                # Update position liquidity based on new capital
                position_liquidity = current_capital * 0.01
            
            # Update portfolio value tracking
            self.portfolio_values.append(current_capital)
        
        # Calculate strategy metrics
        total_fees = sum(self.fees_collected)
        total_il = sum(self.il_incurred)
        total_costs = sum(cost["total_cost"] for cost in self.rebalance_costs) if self.rebalance_costs else 0
        net_return = total_fees - abs(total_il) - total_costs
        
        # Calculate return metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1] if self.portfolio_values else initial_value
        total_return_pct = ((final_value / initial_value) - 1) * 100
        
        # Calculate annualized metrics
        days_elapsed = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']) / SECONDS_PER_DAY
        annualized_return = ((1 + total_return_pct/100) ** (365 / days_elapsed) - 1) * 100 if days_elapsed > 0 else 0
        
        # Calculate additional performance metrics
        drawdown_metrics = self.calculate_drawdown(self.portfolio_values)
        
        # Calculate Sharpe ratio using daily returns
        if self.strategy_returns and np.std(self.strategy_returns) > 0:
            sharpe = np.mean(self.strategy_returns) / np.std(self.strategy_returns) * np.sqrt(365)
        else:
            sharpe = 0
            
        # Calculate Sortino ratio
        sortino = self.calculate_sortino_ratio(self.strategy_returns)
        
        # Calculate Calmar ratio
        calmar = self.calculate_calmar_ratio(annualized_return, drawdown_metrics["max_drawdown_pct"])
        
        # Calculate win/loss metrics
        win_loss_metrics = self.calculate_win_loss_metrics(self.strategy_returns)
        
        # Count triggers by type
        trigger_counts = {}
        for event in self.rebalance_events:
            trigger = event["trigger"]
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        return {
            "rebalance_events": self.rebalance_events,
            "rebalance_count": len(self.rebalance_events),
            "trigger_counts": trigger_counts,
            "initial_capital": self.initial_capital,
            "final_capital": final_value,
            "total_return_pct": total_return_pct,
            "annualized_return": annualized_return,
            "total_fees_collected": total_fees,
            "total_il_incurred": total_il,
            "total_rebalance_costs": total_costs,
            "net_return": net_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": drawdown_metrics["max_drawdown_pct"],
            "max_drawdown_duration": drawdown_metrics["max_drawdown_duration"],
            "recovery_time": drawdown_metrics["recovery_time"],
            "win_rate": win_loss_metrics["win_rate"],
            "profit_factor": win_loss_metrics["profit_factor"],
            "win_loss_ratio": win_loss_metrics["win_loss_ratio"],
            "avg_time_between_rebalances": np.mean([e["time_since_last"] for e in self.rebalance_events[1:]]) / SECONDS_PER_DAY if len(self.rebalance_events) > 1 else 0
        }
    
    def plot_strategy_results(self, results: Dict[str, Any], strategy_name: str = "Adaptive Strategy", output_file: Optional[str] = None) -> None:
        """Plot strategy results."""
        if not self.rebalance_events:
            print("No rebalance events to plot")
            return
        
        # Convert rebalance events to DataFrame
        df_events = pd.DataFrame(self.rebalance_events)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
        
        # Plot 1: Price with rebalance points
        ax1.plot(self.pool_data['datetime'], self.pool_data['price'], color='blue', alpha=0.7)
        
        # Add rebalance points
        for trigger in df_events['trigger'].unique():
            trigger_events = df_events[df_events['trigger'] == trigger]
            ax1.scatter(
                trigger_events['datetime'], 
                trigger_events['price'], 
                label=f'Rebalance ({trigger})',
                marker='o', 
                s=50, 
                alpha=0.7
            )
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'{strategy_name} - Price and Rebalance Events')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Liquidity
        # Scale liquidity to a more readable format (divide by 10^18 to get more manageable numbers)
        scaled_liquidity = self.pool_data['liquidity'].astype(float) / 1e18
        ax2.plot(self.pool_data['datetime'], scaled_liquidity, color='red', alpha=0.7)
        ax2.set_ylabel('Liquidity (×10¹⁸)')
        ax2.set_title('Pool Liquidity')
        ax2.grid(True)
        # Format y-axis to use plain notation instead of scientific notation
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Plot 3: Portfolio Value
        if self.portfolio_values:
            # Create DataFrame for portfolio values
            portfolio_datetimes = pd.to_datetime([self.pool_data.iloc[0]['datetime']] + [e['datetime'] for e in self.rebalance_events])
            
            # If we have more portfolio values than events (tracking between rebalances)
            # then use pool data datetimes
            if len(self.portfolio_values) > len(portfolio_datetimes):
                portfolio_datetimes = self.pool_data['datetime'].iloc[:len(self.portfolio_values)]
            
            # Make sure we have the right number of datetimes
            portfolio_datetimes = portfolio_datetimes[:len(self.portfolio_values)]
            
            df_portfolio = pd.DataFrame({
                'datetime': portfolio_datetimes,
                'value': self.portfolio_values
            })
            
            ax3.plot(df_portfolio['datetime'], df_portfolio['value'], color='green', linewidth=2)
            ax3.set_ylabel('Portfolio Value (USD)')
            ax3.set_title('Portfolio Value Over Time')
            
            # Add horizontal line for initial capital
            if 'initial_capital' in results:
                ax3.axhline(y=results['initial_capital'], color='gray', linestyle='--', alpha=0.7)
                ax3.text(df_portfolio['datetime'].iloc[0], results['initial_capital'], 
                        f"Initial: ${results['initial_capital']:.2f}", verticalalignment='bottom')
            
            # Highlight max drawdown period if available
            if 'max_drawdown_pct' in results and results['max_drawdown_pct'] > 0:
                # Find max drawdown period
                peak = max(self.portfolio_values)
                max_dd_idx = self.portfolio_values.index(min(self.portfolio_values))
                
                # Add annotation
                ax3.annotate(f"Max DD: {results['max_drawdown_pct']:.2f}%", 
                            xy=(df_portfolio['datetime'].iloc[max_dd_idx], self.portfolio_values[max_dd_idx]),
                            xytext=(30, -30),
                            textcoords="offset points",
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            
            ax3.grid(True)
        
        # Plot 4: Cumulative Returns
        if self.strategy_returns:
            cumulative_returns = np.cumsum(self.strategy_returns)
            cumulative_fees = np.cumsum(self.fees_collected)
            cumulative_il = np.cumsum(self.il_incurred)
            
            # Get rebalance costs if available
            if self.rebalance_costs:
                cumulative_costs = np.cumsum([cost["total_cost"] for cost in self.rebalance_costs])
            else:
                cumulative_costs = np.zeros_like(cumulative_returns)
            
            # Check if we have rebalance events to use for datetime
            if self.rebalance_events and len(self.rebalance_events) > 0:
                # Make sure we have enough datetime values for our data
                # If we have more returns than rebalance events, we need to handle this
                if len(cumulative_returns) > len(self.rebalance_events):
                    # Use pool data timestamps instead
                    datetimes = self.pool_data['datetime'].iloc[:len(cumulative_returns)].values
                else:
                    # Use rebalance event datetimes, limited to the length of our returns
                    datetimes = [e['datetime'] for e in self.rebalance_events][:len(cumulative_returns)]
                
                # Create a DataFrame with matching lengths for all arrays
                df_returns = pd.DataFrame({
                    'datetime': datetimes,
                    'cumulative_returns': cumulative_returns,
                    'cumulative_fees': cumulative_fees,
                    'cumulative_il': cumulative_il,
                    'cumulative_costs': cumulative_costs[:len(cumulative_returns)] if len(cumulative_costs) >= len(cumulative_returns) else np.pad(cumulative_costs, (0, len(cumulative_returns) - len(cumulative_costs)))
                })
                
                ax4.plot(df_returns['datetime'], df_returns['cumulative_returns'], 
                        label='Net Returns', color='green', linewidth=2)
                ax4.plot(df_returns['datetime'], df_returns['cumulative_fees'], 
                        label='Fees Collected', color='blue', linestyle='--')
                ax4.plot(df_returns['datetime'], -abs(df_returns['cumulative_il']), 
                        label='Impermanent Loss', color='red', linestyle=':')
                
                if any(cumulative_costs):
                    ax4.plot(df_returns['datetime'], -df_returns['cumulative_costs'], 
                            label='Rebalance Costs', color='orange', linestyle='-.')
                
                ax4.set_ylabel('Cumulative Value (USD)')
                ax4.set_title('Strategy Performance Components')
                ax4.legend()
                ax4.grid(True)
            else:
                # No rebalance events to plot
                ax4.text(0.5, 0.5, 'No rebalance events to plot returns', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes)
                ax4.set_ylabel('Cumulative Value (USD)')
                ax4.set_title('Strategy Performance Components')
                ax4.grid(True)
        
        # Format x-axis
        plt.xlabel('Date')
        date_form = DateFormatter("%Y-%m-%d")
        ax4.xaxis.set_major_formatter(date_form)
        fig.autofmt_xdate()
        
        # Add strategy metrics as text
        metrics_text = (
            f"Initial Capital: ${results.get('initial_capital', 0):.2f}\n"
            f"Final Value: ${results.get('final_capital', 0):.2f}\n"
            f"Total Return: {results.get('total_return_pct', 0):.2f}%\n"
            f"Annualized: {results.get('annualized_return', 0):.2f}%\n\n"
            f"Fees: ${results.get('total_fees_collected', 0):.2f}\n"
            f"IL: ${results.get('total_il_incurred', 0):.2f}\n"
            f"Costs: ${results.get('total_rebalance_costs', 0):.2f}\n\n"
            f"Sharpe: {results.get('sharpe_ratio', 0):.2f}\n"
            f"Sortino: {results.get('sortino_ratio', 0):.2f}\n"
            f"Calmar: {results.get('calmar_ratio', 0):.2f}\n"
            f"Max DD: {results.get('max_drawdown_pct', 0):.2f}%\n\n"
            f"Win Rate: {results.get('win_rate', 0)*100:.1f}%\n"
            f"Profit Factor: {results.get('profit_factor', 0):.2f}\n"
            f"Rebalances: {results.get('rebalance_count', 0)}\n"
            f"Avg Days Between: {results.get('avg_time_between_rebalances', 0):.2f}"
        )
        
        # Move metrics text box to bottom right corner
        fig.text(0.78, 0.02, metrics_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='left')
        
        plt.tight_layout()
        
        # Save the plot if output_file is provided
        if output_file:
            plot_filename = f"{output_file}_plot.png"
            try:
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        else:
            # Use strategy name for the filename
            plot_filename = f"{strategy_name.replace(' ', '_').lower()}_results.png"
            try:
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        
        plt.close(fig)  # Close the figure to free memory
    
    def save_strategy_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save strategy results to CSV and JSON files."""
        print(f"Attempting to save results to files with prefix: {filename}")
        
        # Save rebalance events
        if self.rebalance_events:
            rebalance_file = f"{filename}_rebalance_events.csv"
            print(f"Saving rebalance events to: {rebalance_file}")
            df_events = pd.DataFrame(self.rebalance_events)
            df_events.to_csv(rebalance_file, index=False)
            print(f"Rebalance events saved to {rebalance_file}")
        else:
            print("No rebalance events to save")
        
        # Save metrics
        metrics_file = f"{filename}_metrics.json"
        print(f"Saving metrics to: {metrics_file}")
        metrics = {k: v for k, v in results.items() if k != "rebalance_events"}
        
        # Convert numpy values to Python native types for JSON serialization
        for k, v in metrics.items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                metrics[k] = int(v)
            elif isinstance(v, (np.float64, np.float32, np.float16)):
                metrics[k] = float(v)
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Strategy metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Error saving metrics: {e}")


def main():
    """Main function to run the strategy backtesting."""
    parser = argparse.ArgumentParser(description="Backtest Uniswap V3 liquidity provision strategies")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_FILE,
                        help=f"Pool data CSV file (default: {DEFAULT_DATA_FILE})")
    parser.add_argument("--il", type=str, default=DEFAULT_IL_FILE,
                        help=f"Impermanent loss data CSV file (default: {DEFAULT_IL_FILE})")
    parser.add_argument("--tick-range", type=int, default=10,
                        help="Tick range for liquidity provision (default: 10)")
    parser.add_argument("--interval", type=str, default="daily",
                        help="Rebalance interval: 'hourly', 'daily', 'weekly', or seconds (default: 'daily')")
    parser.add_argument("--price-threshold", type=float, default=0.5,
                        help="Price change threshold for rebalancing in percent (default: 0.5)")
    parser.add_argument("--volatility-threshold", type=float, default=1.0,
                        help="Volatility threshold for rebalancing (default: 1.0)")
    parser.add_argument("--output", type=str, default="strategy_results",
                        help="Base name for output files (default: strategy_results)")
    parser.add_argument("--strategy", type=str, default="adaptive",
                        help="Strategy type: 'time' or 'adaptive' (default: 'adaptive')")
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                        help="Initial capital in USD (default: 100000.0)")
    parser.add_argument("--daily-volume", type=float, default=None,
                        help="Average daily trading volume in USD for more accurate fee calculation")
    parser.add_argument("--tick-data", type=str, default=None,
                        help="Path to tick data JSON file mapping block numbers to tick liquidity data")
    parser.add_argument("--gas-price", type=float, default=30.0,
                        help="Gas price in Gwei for transaction cost calculation (default: 30.0)")
    parser.add_argument("--compare-strategies", action="store_true",
                        help="Compare multiple strategies with different parameters")
    
    args = parser.parse_args()
    
    # Convert interval string to seconds
    if args.interval == "hourly":
        interval_seconds = 3600
    elif args.interval == "daily":
        interval_seconds = SECONDS_PER_DAY
    elif args.interval == "weekly":
        interval_seconds = SECONDS_PER_WEEK
    else:
        try:
            interval_seconds = int(args.interval)
        except ValueError:
            print(f"Invalid interval: {args.interval}. Using daily.")
            interval_seconds = SECONDS_PER_DAY
    
    # Load tick data if provided
    tick_data = None
    if args.tick_data and os.path.exists(args.tick_data):
        try:
            with open(args.tick_data, 'r') as f:
                tick_data = json.load(f)
            print(f"Loaded tick data from {args.tick_data}")
        except Exception as e:
            print(f"Error loading tick data: {e}")
    
    # Initialize strategy
    strategy = LiquidityProvisionStrategy(args.data, args.il)
    
    # Set strategy parameters
    strategy.set_strategy_params(
        tick_range=args.tick_range,
        rebalance_interval=interval_seconds,
        price_threshold=args.price_threshold,
        volatility_threshold=args.volatility_threshold,
        initial_capital=args.initial_capital
    )
    
    # If comparing strategies, run multiple backtests with different parameters
    if args.compare_strategies:
        print("Comparing multiple strategies...")
        
        # Define strategies to compare
        strategies = [
            {
                "name": "Conservative",
                "tick_range": 20,
                "price_threshold": 1.0,
                "volatility_threshold": 2.0,
                "interval": SECONDS_PER_DAY * 2  # Every 2 days
            },
            {
                "name": "Moderate",
                "tick_range": 10,
                "price_threshold": 0.5,
                "volatility_threshold": 1.0,
                "interval": SECONDS_PER_DAY  # Daily
            },
            {
                "name": "Aggressive",
                "tick_range": 5,
                "price_threshold": 0.25,
                "volatility_threshold": 0.5,
                "interval": SECONDS_PER_DAY / 2  # Twice daily
            }
        ]
        
        results = []
        
        for strat in strategies:
            print(f"\nTesting {strat['name']} strategy...")
            strategy.set_strategy_params(
                tick_range=strat["tick_range"],
                rebalance_interval=strat["interval"],
                price_threshold=strat["price_threshold"],
                volatility_threshold=strat["volatility_threshold"],
                initial_capital=args.initial_capital
            )
            
            # Run backtest with strategy parameters
            result = strategy.backtest_adaptive_strategy(
                daily_volume=args.daily_volume,
                tick_data=tick_data
            )
            
            # Add strategy name
            result["strategy_name"] = strat["name"]
            results.append(result)
            
            # Plot and save results
            strategy.plot_strategy_results(
                result, 
                f"{strat['name']} Strategy", 
                f"{args.output}_{strat['name'].lower()}"
            )
            
            # Save results
            strategy.save_strategy_results(
                result, 
                f"{args.output}_{strat['name'].lower()}"
            )
        
        # Compare and print results
        print("\nStrategy Comparison:")
        print("-" * 80)
        print(f"{'Strategy':<15} {'Return %':<10} {'Sharpe':<8} {'Sortino':<8} {'Max DD %':<10} {'Win Rate':<10} {'Profit Factor':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['strategy_name']:<15} "
                  f"{result['total_return_pct']:.2f}%     "
                  f"{result['sharpe_ratio']:.2f}    "
                  f"{result['sortino_ratio']:.2f}    "
                  f"{result['max_drawdown_pct']:.2f}%      "
                  f"{result['win_rate']*100:.1f}%      "
                  f"{result['profit_factor']:.2f}")
        
        print("-" * 80)
        
    else:
        # Run single backtest
        if args.strategy == "time":
            print("Running time-based strategy backtest...")
            results = strategy.backtest_time_based_strategy()
            strategy_name = "Time-Based Strategy"
        else:
            print("Running adaptive strategy backtest...")
            results = strategy.backtest_adaptive_strategy(
                daily_volume=args.daily_volume,
                tick_data=tick_data
            )
            strategy_name = "Adaptive Strategy"
        
        # Print results
        print("\nStrategy Backtesting Results:")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Total Rebalances: {results['rebalance_count']}")
        print(f"Total Fees Collected: ${results['total_fees_collected']:.2f}")
        print(f"Total IL Incurred: ${results['total_il_incurred']:.2f}")
        print(f"Total Rebalance Costs: ${results.get('total_rebalance_costs', 0):.2f}")
        print(f"Net Return: ${results['net_return']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {results.get('sortino_ratio', 0):.4f}")
        print(f"Calmar Ratio: {results.get('calmar_ratio', 0):.4f}")
        print(f"Maximum Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        print(f"Win Rate: {results.get('win_rate', 0)*100:.1f}%")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Win/Loss Ratio: {results.get('win_loss_ratio', 0):.2f}")
        print(f"Avg Days Between Rebalances: {results['avg_time_between_rebalances']:.2f}")
        
        if args.strategy == "adaptive" and "trigger_counts" in results:
            print("\nRebalance Triggers:")
            for trigger, count in results["trigger_counts"].items():
                print(f"- {trigger}: {count} ({count/results['rebalance_count']*100:.1f}%)")
        
        # Plot results
        strategy.plot_strategy_results(results, strategy_name, args.output)
        
        # Save results
        strategy.save_strategy_results(results, args.output)
        print(f"Results saved with prefix: {args.output}")


if __name__ == "__main__":
    main() 