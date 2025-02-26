"""
Uniswap V3 Pool Data Collector

This script collects comprehensive historical data from a Uniswap V3 pool
for backtesting dynamic liquidity provision strategies and analyzing metrics
such as impermanent loss (IL), Sharpe ratio, and other performance indicators.

Usage:
    python pool_data.py --pool <pool_address> --days <days_to_collect> --output <output_file>
"""

import os
import json
import time
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from web3 import Web3
from web3.middleware import geth_poa_middleware
from multicall import Call, Multicall
import random
from requests.exceptions import RequestException, Timeout
from abi import POOL_ABI, ERC20_ABI

# Load environment variables
load_dotenv()

# Configuration
PRIMARY_RPC_URL = f"https://eth-mainnet.g.alchemy.com/v2/{os.getenv('ALCHEMY_API_KEY')}"
# List of public RPCs for failover
PUBLIC_RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://eth-mainnet.public.blastapi.io",
    "https://rpc.flashbots.net",
    "https://eth.blockrazor.xyz",
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.mevblocker.io",
    "https://rpc.ankr.com/eth",
    "https://go.getblock.io/aefd01aa907c4805ba3c00a9e5b48c6b",
    "https://eth-mainnet.nodereal.io/v1/1659dfb40aa24bbb8153a677b98064d7",
    "https://api.securerpc.com/v1",
    "https://eth.merkle.io",
    "https://eth.blockrazor.xyz"
]

DEFAULT_POOL_ADDRESS = "0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa"  # wstETH/ETH 0.01%
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BLOCK_TIME_SECONDS = 12  # Average Ethereum block time
BLOCKS_PER_DAY = 24 * 60 * 60 // BLOCK_TIME_SECONDS  # ~7200 blocks per day
DATA_DIR = Path("data")

# Load configuration from .env
DEFAULT_BLOCK_STEP = int(os.getenv("BLOCK_STEP", "1000"))
ENABLE_ADAPTIVE_SAMPLING = os.getenv("ENABLE_ADAPTIVE_SAMPLING", "true").lower() == "true"
ADAPTIVE_SAMPLING_THRESHOLD = float(os.getenv("ADAPTIVE_SAMPLING_THRESHOLD", "5.0"))
ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

class RPCManager:
    """
    Manager for Ethereum RPC connections with failover and retry capabilities.
    Handles RPC errors, rate limits, and timeouts by switching between available RPCs.
    """
    
    def __init__(self, primary_rpc_url: str = PRIMARY_RPC_URL, public_rpc_urls: List[str] = PUBLIC_RPC_URLS):
        """
        Initialize the RPC manager with a primary RPC URL and a list of public fallback RPCs.
        
        Args:
            primary_rpc_url: The primary RPC endpoint to use (typically Infura or Alchemy)
            public_rpc_urls: List of public RPC endpoints to use as fallbacks
        """
        self.primary_rpc_url = primary_rpc_url
        self.public_rpc_urls = public_rpc_urls
        self.current_rpc_url = primary_rpc_url
        
        # Track RPC availability and rate limiting
        self.rpc_status = {self.primary_rpc_url: {"available": True, "failures": 0, "last_failure": 0}}
        for url in self.public_rpc_urls:
            self.rpc_status[url] = {"available": True, "failures": 0, "last_failure": 0}
        
        # Initialize Web3 with primary RPC
        self.web3 = self._create_web3_instance(self.primary_rpc_url)
        
        # Last successful RPC
        self.last_successful_rpc = self.primary_rpc_url
        
        print(f"RPC Manager initialized with primary RPC and {len(public_rpc_urls)} fallback RPCs")
    
    def _create_web3_instance(self, rpc_url: str) -> Web3:
        """Create a Web3 instance with the given RPC URL."""
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        
        # Add middleware for better compatibility
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        return w3
    
    def _is_rate_limited(self, error: Exception) -> bool:
        """Check if an error is related to rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit", "exceeded", "too many requests", "429", 
            "limit reached", "throttled", "quota exceeded"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def _mark_rpc_unavailable(self, rpc_url: str, error: Exception = None) -> None:
        """Mark an RPC as temporarily unavailable."""
        self.rpc_status[rpc_url]["available"] = False
        self.rpc_status[rpc_url]["failures"] += 1
        self.rpc_status[rpc_url]["last_failure"] = time.time()
        
        # If it's rate limited, mark it for a longer cooldown
        if error and self._is_rate_limited(error):
            # For rate limits, we'll use a longer cooldown (5 minutes)
            self.rpc_status[rpc_url]["cooldown_until"] = time.time() + 300
            print(f"RPC {rpc_url} rate limited, cooling down for 5 minutes")
        else:
            # For other errors, shorter cooldown (30 seconds)
            self.rpc_status[rpc_url]["cooldown_until"] = time.time() + 30
            if error:
                print(f"RPC {rpc_url} failed with error: {str(error)[:100]}...")
    
    def _select_next_available_rpc(self) -> str:
        """Select the next available RPC from the list."""
        current_time = time.time()
        
        # Check if primary RPC has recovered
        if self.current_rpc_url != self.primary_rpc_url:
            primary_status = self.rpc_status[self.primary_rpc_url]
            if primary_status["available"] or (
                "cooldown_until" in primary_status and 
                current_time > primary_status["cooldown_until"]
            ):
                self.rpc_status[self.primary_rpc_url]["available"] = True
                print(f"Returning to primary RPC: {self.primary_rpc_url}")
                return self.primary_rpc_url
        
        # Try to use the last successful RPC first if it's not the current one
        if (self.last_successful_rpc != self.current_rpc_url and 
            (self.rpc_status[self.last_successful_rpc]["available"] or 
             ("cooldown_until" in self.rpc_status[self.last_successful_rpc] and 
              current_time > self.rpc_status[self.last_successful_rpc]["cooldown_until"]))):
            self.rpc_status[self.last_successful_rpc]["available"] = True
            return self.last_successful_rpc
        
        # Otherwise, find an available public RPC
        available_rpcs = []
        for url in self.public_rpc_urls:
            status = self.rpc_status[url]
            # Check if available or cooldown period has passed
            if status["available"] or (
                "cooldown_until" in status and 
                current_time > status["cooldown_until"]
            ):
                self.rpc_status[url]["available"] = True
                available_rpcs.append(url)
        
        # If we have available RPCs, choose one (preferring less-used ones)
        if available_rpcs:
            # Sort by failure count (ascending)
            available_rpcs.sort(key=lambda url: self.rpc_status[url]["failures"])
            # Choose one of the top 3 with least failures, with some randomness
            return random.choice(available_rpcs[:min(3, len(available_rpcs))])
        
        # If no RPCs are available, wait for the one with the shortest cooldown
        cooldown_rpcs = {
            url: status["cooldown_until"] 
            for url, status in self.rpc_status.items() 
            if "cooldown_until" in status
        }
        
        if cooldown_rpcs:
            # Find RPC with the shortest cooldown
            next_available = min(cooldown_rpcs.items(), key=lambda x: x[1])
            wait_time = max(0, next_available[1] - current_time)
            
            if wait_time > 0:
                print(f"All RPCs are rate limited. Waiting {wait_time:.1f} seconds for next available RPC...")
                time.sleep(wait_time)
            
            self.rpc_status[next_available[0]]["available"] = True
            return next_available[0]
        
        # Last resort: just return primary and hope for the best
        print("Warning: No available RPCs found. Falling back to primary RPC.")
        return self.primary_rpc_url
    
    def switch_rpc(self, error: Exception = None) -> bool:
        """
        Switch to a different RPC provider.
        
        Args:
            error: The exception that caused the switch, if any
            
        Returns:
            bool: True if switched successfully, False otherwise
        """
        # Mark current RPC as unavailable
        self._mark_rpc_unavailable(self.current_rpc_url, error)
        
        # Select a new RPC
        new_rpc_url = self._select_next_available_rpc()
        
        if new_rpc_url == self.current_rpc_url:
            print(f"No alternative RPCs available, staying with current RPC: {self.current_rpc_url}")
            return False
        
        # Update current RPC and Web3 instance
        self.current_rpc_url = new_rpc_url
        self.web3 = self._create_web3_instance(new_rpc_url)
        
        print(f"Switched to RPC: {new_rpc_url}")
        return True
    
    def execute_with_retry(self, func: Callable, *args, max_retries: int = 2, **kwargs) -> Any:
        """
        Execute a function with automatic RPC failover and retry logic.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            max_retries: Maximum number of retries per RPC
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        
        Raises:
            Exception: If all retries and RPC failovers are exhausted
        """
        total_attempts = 0
        max_total_attempts = (len(self.public_rpc_urls) + 1) * (max_retries + 1)
        errors = []
        
        while total_attempts < max_total_attempts:
            rpc_url = self.current_rpc_url
            retries = 0
            
            while retries <= max_retries:
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Mark this RPC as successful
                    self.last_successful_rpc = self.current_rpc_url
                    return result
                    
                except (RequestException, Timeout) as e:
                    # Network-related exceptions
                    errors.append(f"Network error with {rpc_url}: {str(e)}")
                    retry_or_switch = True
                    
                except ValueError as e:
                    error_str = str(e).lower()
                    if "json" in error_str or "response" in error_str or "connect" in error_str:
                        # RPC response parsing errors
                        errors.append(f"RPC error with {rpc_url}: {str(e)}")
                        retry_or_switch = True
                    else:
                        # Value errors that are likely code issues, not RPC issues
                        raise
                        
                except Exception as e:
                    # Other exceptions
                    errors.append(f"General error with {rpc_url}: {str(e)}")
                    
                    # Check if it's a rate limiting issue
                    if self._is_rate_limited(e):
                        retry_or_switch = True
                    else:
                        # For other errors, retry but with less confidence
                        retry_or_switch = retries < max_retries
                
                # Increment counters
                retries += 1
                total_attempts += 1
                
                # Decide whether to retry with current RPC or switch
                if retries > max_retries or self._is_rate_limited(errors[-1]):
                    # If we've exhausted retries or hit rate limits, try to switch RPC
                    if self.switch_rpc(Exception(errors[-1])):
                        # Successfully switched, break retry loop to start fresh with new RPC
                        break
                else:
                    # Retry with exponential backoff
                    wait_time = RETRY_DELAY * (2 ** (retries - 1))
                    print(f"Retrying with {rpc_url} in {wait_time:.1f} seconds (attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
        
        # If we reach here, all RPCs and retries have been exhausted
        error_summary = "\n".join(errors[-5:])  # Show the last 5 errors
        raise Exception(f"Failed after {total_attempts} attempts across multiple RPCs:\n{error_summary}")
    
    def execute_multicall(self, calls: List[Call], block_id: int = None) -> Dict[str, Any]:
        """
        Execute a multicall with robust error handling and automatic RPC failover.
        
        Args:
            calls: List of multicall.Call objects to execute
            block_id: Optional block identifier for historical queries
            
        Returns:
            Dict containing the multicall results
        """
        def _do_multicall():
            """Execute the multicall using the current Web3 instance."""
            multicall = Multicall(calls, _w3=self.web3, block_id=block_id)
            return multicall()
        
        return self.execute_with_retry(_do_multicall)
    
    def get_contract(self, address: str, abi: List[Dict[str, Any]]) -> Any:
        """Get a contract instance using the current Web3 provider."""
        return self.web3.eth.contract(address=address, abi=abi)
    
    def call_contract_function(self, contract_func, block_identifier: int = None) -> Any:
        """
        Call a contract function with retry logic and RPC failover.
        
        Args:
            contract_func: The contract function to call
            block_identifier: Optional block number for historical queries
            
        Returns:
            The result of the contract function call
        """
        def _do_call():
            """Execute the contract call."""
            if block_identifier is not None:
                return contract_func.call(block_identifier=block_identifier)
            else:
                return contract_func.call()
        
        return self.execute_with_retry(_do_call)
    
    def get_block(self, block_number: int) -> Any:
        """
        Get block information with retry logic and RPC failover.
        
        Args:
            block_number: The block number to retrieve
            
        Returns:
            The block information
        """
        def _do_get_block():
            """Execute the get_block call."""
            return self.web3.eth.get_block(block_number)
        
        return self.execute_with_retry(_do_get_block)


# Initialize global RPC manager with primary RPC
rpc_manager = RPCManager(primary_rpc_url=PRIMARY_RPC_URL, public_rpc_urls=PUBLIC_RPC_URLS)

# Initialize Web3 with the RPC manager's web3 instance
web3 = rpc_manager.web3

class UniswapV3PoolDataCollector:
    """Class for collecting and analyzing Uniswap V3 pool data."""
    
    def __init__(self, pool_address: str, rpc_manager: RPCManager = rpc_manager, max_cache_size: int = 10000):
        """Initialize with pool address and RPC URL."""
        self.pool_address = Web3.to_checksum_address(pool_address)
        self.rpc_manager = rpc_manager
        self.web3 = rpc_manager.web3
        self.pool_contract = rpc_manager.get_contract(self.pool_address, POOL_ABI)
        
        # Get pool metadata
        self.token0_address = self.rpc_manager.call_contract_function(self.pool_contract.functions.token0())
        self.token1_address = self.rpc_manager.call_contract_function(self.pool_contract.functions.token1())
        self.fee = self.rpc_manager.call_contract_function(self.pool_contract.functions.fee())
        self.tick_spacing = self.rpc_manager.call_contract_function(self.pool_contract.functions.tickSpacing())
        
        # Get token metadata
        self.token0_contract = self.rpc_manager.get_contract(self.token0_address, ERC20_ABI)
        self.token1_contract = self.rpc_manager.get_contract(self.token1_address, ERC20_ABI)
        
        self.token0_symbol = self.rpc_manager.call_contract_function(self.token0_contract.functions.symbol())
        self.token1_symbol = self.rpc_manager.call_contract_function(self.token1_contract.functions.symbol())
        self.token0_decimals = self.rpc_manager.call_contract_function(self.token0_contract.functions.decimals())
        self.token1_decimals = self.rpc_manager.call_contract_function(self.token1_contract.functions.decimals())
        
        # Cache blocks to timestamps mapping with size limit
        self.block_timestamps = {}
        self.max_cache_size = max_cache_size
        
        # Add a multicall result cache
        self.multicall_cache = {}
        
        # Track optimal batch sizes for different call types
        self.batch_size_history = {
            "ticks": {},
            "pool_data": {},
            "general": {}
        }
        
        # Initialize batch size tracking
        self.tick_batch_size_history = {}
        
        print(f"Initialized data collector for {self.token0_symbol}/{self.token1_symbol} "
              f"{self.fee/10000}% pool ({self.pool_address})")
    
    def track_multicall_performance(self, call_type, batch_size, success, execution_time):
        """Track multicall performance to optimize batch sizes."""
        if call_type not in self.batch_size_history:
            self.batch_size_history[call_type] = {}
            
        if batch_size not in self.batch_size_history[call_type]:
            self.batch_size_history[call_type][batch_size] = {
                "success_count": 0,
                "failure_count": 0,
                "total_time": 0,
                "avg_time": 0
            }
            
        stats = self.batch_size_history[call_type][batch_size]
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
            
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / (stats["success_count"] + stats["failure_count"])
    
    def get_optimal_batch_size(self, pool_address, call_type):
        """Get the optimal batch size based on historical performance."""
        # Start with a conservative default for ticks
        if call_type == "ticks":
            # Use a smaller default batch size for ticks (5 instead of 10)
            default_size = 5
            return self.tick_batch_size_history.get(pool_address, default_size)
        elif call_type in self.batch_size_history:
            # Find the batch size with the highest success rate and lowest average time
            best_batch_size = 10  # Default
            best_score = 0
            
            for batch_size, stats in self.batch_size_history[call_type].items():
                total_calls = stats["success_count"] + stats["failure_count"]
                if total_calls == 0:
                    continue
                    
                success_rate = stats["success_count"] / total_calls
                # Penalize larger batch sizes slightly to prefer smaller ones when close
                size_penalty = 1.0 - (batch_size / 100)  # Small penalty for larger sizes
                
                # Score based on success rate and average time (lower is better)
                if stats["avg_time"] > 0:
                    time_factor = 1.0 / stats["avg_time"]
                else:
                    time_factor = 1.0
                    
                score = success_rate * time_factor * size_penalty
                
                if score > best_score:
                    best_score = score
                    best_batch_size = batch_size
                    
            return best_batch_size
        
        return 10  # Default batch size
    
    def get_block_timestamp(self, block_number: int) -> int:
        """Get the timestamp for a block number, with caching."""
        if block_number not in self.block_timestamps:
            try:
                block = self.rpc_manager.get_block(block_number)
                self.block_timestamps[block_number] = block.timestamp
                
                # Limit cache size to prevent memory issues with large datasets
                if len(self.block_timestamps) > self.max_cache_size:
                    # Remove oldest 10% of entries when limit is reached
                    remove_count = self.max_cache_size // 10
                    oldest_blocks = sorted(self.block_timestamps.keys())[:remove_count]
                    for old_block in oldest_blocks:
                        del self.block_timestamps[old_block]
                        
            except Exception as e:
                print(f"Error getting timestamp for block {block_number}: {e}")
                return None
        return self.block_timestamps[block_number]
    
    def sqrt_price_x96_to_price(self, sqrt_price_x96: int) -> float:
        """Convert sqrtPriceX96 to human-readable price."""
        # Uniswap stores sqrt(price) * 2^96
        price = (sqrt_price_x96 / 2**96) ** 2
        
        # Adjust for token decimals
        price_adjusted = price * (10 ** (self.token1_decimals - self.token0_decimals))
        return price_adjusted
    
    def tick_to_price(self, tick: int) -> float:
        """Convert tick to price."""
        return 1.0001 ** tick * (10 ** (self.token1_decimals - self.token0_decimals))
    
    def get_pool_data_at_block(self, block_number: int, with_retries: bool = True) -> Optional[Dict[str, Any]]:
        """Get pool data at a specific block with retry logic and multicall optimization."""
        try:
            # Get timestamp first (not included in multicall)
            timestamp = self.get_block_timestamp(block_number)
            
            if timestamp is None:
                print(f"Skipping block {block_number} - unable to get timestamp")
                return None
            
            # Setup multicall
            calls = [
                Call(
                    "0x0000000000000000000000000000000000000000",
                    ['getBlockByNumber(uint256)(uint256)', block_number],
                    [['timestamp', None]]
                ),
                Call(
                    self.pool_address,
                    ['slot0()(uint160,int24,uint16,uint16,uint16,uint8,bool)'],
                    [['sqrtPriceX96', None], ['tick', None], ['observationIndex', None],
                     ['observationCardinality', None], ['observationCardinalityNext', None],
                     ['feeProtocol', None], ['unlocked', None]]
                ),
                Call(
                    self.pool_address,
                    ['liquidity()(uint128)'],
                    [['liquidity', None]]
                )
            ]
            
            # Use cached multicall instead of direct execution
            try:
                results = self.execute_multicall_with_cache(calls, block_number)
                
                # More robust error checking for multicall results
                if results is None or not isinstance(results, dict):
                    raise Exception(f"Invalid multicall result type: {type(results)}")
                
                # Verify all expected keys are in the results
                expected_keys = ['sqrtPriceX96', 'tick', 'observationIndex', 'observationCardinality', 
                                'observationCardinalityNext', 'feeProtocol', 'unlocked', 'liquidity']
                
                missing_keys = [key for key in expected_keys if key not in results]
                if missing_keys:
                    raise Exception(f"Missing keys in multicall results: {missing_keys}")
                
                # Extract values from results with better type checking
                sqrt_price_x96 = int(results.get('sqrtPriceX96', 0))
                tick = int(results.get('tick', 0))
                observation_index = int(results.get('observationIndex', 0))
                observation_cardinality = int(results.get('observationCardinality', 0))
                observation_cardinality_next = int(results.get('observationCardinalityNext', 0))
                fee_protocol = int(results.get('feeProtocol', 0))
                unlocked = bool(results.get('unlocked', False))
                liquidity = int(results.get('liquidity', 0))
                
                # Convert sqrtPriceX96 to price
                price = self.sqrt_price_x96_to_price(sqrt_price_x96)
                
                return {
                    "block": block_number,
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "sqrtPriceX96": sqrt_price_x96,
                    "tick": tick,
                    "price": price,
                    "liquidity": liquidity,
                    "observationIndex": observation_index,
                    "observationCardinality": observation_cardinality,
                    "observationCardinalityNext": observation_cardinality_next,
                    "feeProtocol": fee_protocol,
                    "unlocked": unlocked
                }
            
            except Exception as e:
                # If multicall fails, fall back to direct calls
                print(f"Multicall failed at block {block_number}, falling back to direct calls: {e}")
                
                # Get slot0 data using RPC manager
                slot0_result = self.rpc_manager.call_contract_function(
                    self.pool_contract.functions.slot0(), 
                    block_identifier=block_number
                )
                
                sqrt_price_x96 = slot0_result[0]
                tick = slot0_result[1]
                observation_index = slot0_result[2]
                observation_cardinality = slot0_result[3]
                observation_cardinality_next = slot0_result[4]
                fee_protocol = slot0_result[5]
                unlocked = slot0_result[6]
                
                # Get liquidity using RPC manager
                liquidity = self.rpc_manager.call_contract_function(
                    self.pool_contract.functions.liquidity(),
                    block_identifier=block_number
                )
                
                # Convert sqrtPriceX96 to price
                price = self.sqrt_price_x96_to_price(sqrt_price_x96)
                
                return {
                    "block": block_number,
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    "sqrtPriceX96": sqrt_price_x96,
                    "tick": tick,
                    "price": price,
                    "liquidity": liquidity,
                    "observationIndex": observation_index,
                    "observationCardinality": observation_cardinality,
                    "observationCardinalityNext": observation_cardinality_next,
                    "feeProtocol": fee_protocol,
                    "unlocked": unlocked
                }
        
        except Exception as e:
            print(f"Failed to get pool data at block {block_number}: {e}")
            return None
    
    def get_active_ticks_at_block(self, block_number: int, tick_range: int = 20) -> List[Dict[str, Any]]:
        """Get active ticks around the current price at a specific block using multicall."""
        try:
            # Get current tick first - this is important to avoid multicall issues
            try:
                # Use cached pool data if available
                cache_key = f"pool_data_{block_number}"
                if cache_key in self.multicall_cache:
                    pool_data = self.multicall_cache[cache_key]
                    current_tick = pool_data.get("tick")
                else:
                    # Get current tick directly
                    slot0 = self.rpc_manager.call_contract_function(
                        self.pool_contract.functions.slot0(),
                        block_identifier=block_number
                    )
                    current_tick = slot0[1]
            except Exception as e:
                print(f"Error getting current tick: {e}")
                return []
            
            # Define range of ticks to check
            tick_spacing = self.tick_spacing
            lower_tick = current_tick - (tick_range * tick_spacing)
            upper_tick = current_tick + (tick_range * tick_spacing)
            
            # Generate potential tick values
            potential_ticks = []
            tick = lower_tick - (lower_tick % tick_spacing)
            while tick <= upper_tick:
                potential_ticks.append(tick)
                tick += tick_spacing
            
            # Get optimal batch size based on historical performance
            batch_size = self.get_optimal_batch_size(self.pool_address, "ticks")
            # print(f"Checking ticks from {lower_tick} to {upper_tick} with spacing {tick_spacing} (batch size: {batch_size})")
            
            # Process ticks in batches to avoid gas limit errors
            active_ticks = []
            
            def process_batch_results(results, batch_ticks):
                """Helper function to process batch results consistently"""
                batch_active_ticks = []
                
                if 'initialized' in results and isinstance(results['initialized'], list):
                    # Handle case where results are returned as lists
                    for idx, tick in enumerate(batch_ticks):
                        if idx < len(results['initialized']) and results['initialized'][idx]:
                            batch_active_ticks.append({
                                "block": block_number,
                                "tick": tick,
                                "liquidityGross": results['liquidityGross'][idx] if 'liquidityGross' in results else 0,
                                "liquidityNet": results['liquidityNet'][idx] if 'liquidityNet' in results else 0,
                            })
                elif 'initialized' in results:
                    # Handle case where there's a single result
                    if results['initialized'] and batch_ticks:
                        batch_active_ticks.append({
                            "block": block_number,
                            "tick": batch_ticks[0],
                            "liquidityGross": results.get('liquidityGross', 0),
                            "liquidityNet": results.get('liquidityNet', 0),
                        })
                
                return batch_active_ticks
            
            for i in range(0, len(potential_ticks), batch_size):
                batch_ticks = potential_ticks[i:i+batch_size]
                
                # Create multicall for this batch
                calls = []
                for tick in batch_ticks:
                    calls.append(
                        Call(
                            self.pool_address,
                            ['ticks(int24)(uint128,int128,uint256,uint256,int56,uint160,uint32,bool)',
                             tick],
                            [['liquidityGross', None], ['liquidityNet', None], ['feeGrowthOutside0X128', None],
                             ['feeGrowthOutside1X128', None], ['tickCumulativeOutside', None],
                             ['secondsPerLiquidityOutsideX128', None], ['secondsOutside', None],
                             ['initialized', None]]
                        )
                    )
                
                # Use cached multicall with unique key for this batch
                cache_key = f"ticks_{block_number}_{i}_{batch_size}"
                try:
                    if cache_key in self.multicall_cache:
                        results = self.multicall_cache[cache_key]
                    else:
                        results = self.rpc_manager.execute_multicall(calls, block_id=block_number)
                        self.multicall_cache[cache_key] = results
                    
                    # Update batch size history based on success
                    self.update_batch_size_history(self.pool_address, "ticks", True, batch_size)
                    
                    # Process results
                    batch_active_ticks = process_batch_results(results, batch_ticks)
                    active_ticks.extend(batch_active_ticks)
                    
                    # If multicall didn't work as expected, fall back to direct calls
                    if not batch_active_ticks and 'initialized' not in results:
                        print(f"Unexpected multicall result format. Falling back to direct calls.")
                        for tick in batch_ticks:
                            try:
                                tick_data = self.rpc_manager.call_contract_function(
                                    self.pool_contract.functions.ticks(tick),
                                    block_identifier=block_number
                                )
                                
                                initialized = tick_data[7]
                                if initialized:
                                    active_ticks.append({
                                        "block": block_number,
                                        "tick": tick,
                                        "liquidityGross": tick_data[0],
                                        "liquidityNet": tick_data[1],
                                    })
                            except Exception as tick_e:
                                # Silent continue if a single tick fails
                                continue
                
                except Exception as e:
                    print(f"Multicall failed: {e}")
                    # Update batch size history based on failure
                    self.update_batch_size_history(self.pool_address, "ticks", False, batch_size)
                    
                    # Handle gas limit errors by retrying with smaller batches
                    if "gas required exceeds allowance" in str(e) or "execution reverted" in str(e):
                        print(f"Gas limit error with batch size {batch_size}, retrying with smaller batches")
                        # Try with smaller sub-batches
                        sub_batch_size = max(1, batch_size // 2)
                        for j in range(0, len(batch_ticks), sub_batch_size):
                            sub_batch = batch_ticks[j:j+sub_batch_size]
                            sub_calls = []
                            for tick in sub_batch:
                                sub_calls.append(
                                    Call(
                                        self.pool_address,
                                        ['ticks(int24)(uint128,int128,uint256,uint256,int56,uint160,uint32,bool)',
                                         tick],
                                        [['liquidityGross', None], ['liquidityNet', None], 
                                         ['feeGrowthOutside0X128', None], ['feeGrowthOutside1X128', None],
                                         ['tickCumulativeOutside', None], ['secondsPerLiquidityOutsideX128', None],
                                         ['secondsOutside', None], ['initialized', None]]
                                    )
                                )
                            
                            # Use cached multicall for sub-batch
                            sub_cache_key = f"ticks_{block_number}_{i}_{j}_{sub_batch_size}"
                            try:
                                if sub_cache_key in self.multicall_cache:
                                    sub_results = self.multicall_cache[sub_cache_key]
                                else:
                                    sub_results = self.rpc_manager.execute_multicall(sub_calls, block_id=block_number)
                                    self.multicall_cache[sub_cache_key] = sub_results
                                
                                # Process sub-batch results
                                batch_active_ticks = process_batch_results(sub_results, sub_batch)
                                active_ticks.extend(batch_active_ticks)
                                
                                # If multicall didn't work, fall back to direct calls
                                if not batch_active_ticks and 'initialized' not in sub_results:
                                    # Fall back to direct calls
                                    for tick in sub_batch:
                                        try:
                                            tick_data = self.rpc_manager.call_contract_function(
                                                self.pool_contract.functions.ticks(tick),
                                                block_identifier=block_number
                                            )
                                            
                                            initialized = tick_data[7]
                                            if initialized:
                                                active_ticks.append({
                                                    "block": block_number,
                                                    "tick": tick,
                                                    "liquidityGross": tick_data[0],
                                                    "liquidityNet": tick_data[1],
                                                })
                                        except Exception:
                                            # Silent continue if a single tick fails
                                            continue
                            except Exception as sub_e:
                                print(f"Sub-batch multicall failed: {sub_e}. Falling back to direct calls.")
                                # Fall back to direct calls for this sub-batch
                                for tick in sub_batch:
                                    try:
                                        tick_data = self.rpc_manager.call_contract_function(
                                            self.pool_contract.functions.ticks(tick),
                                            block_identifier=block_number
                                        )
                                        
                                        initialized = tick_data[7]
                                        if initialized:
                                            active_ticks.append({
                                                "block": block_number,
                                                "tick": tick,
                                                "liquidityGross": tick_data[0],
                                                "liquidityNet": tick_data[1],
                                            })
                                    except Exception:
                                        # Silent continue if a single tick fails
                                        continue
                    else:
                        # Fall back to direct calls if not a gas limit error or batch size is already small
                        print(f"Falling back to direct calls for batch")
                        for tick in batch_ticks:
                            try:
                                # Use RPC manager for direct calls
                                tick_data = self.rpc_manager.call_contract_function(
                                    self.pool_contract.functions.ticks(tick),
                                    block_identifier=block_number
                                )
                                
                                initialized = tick_data[7]
                                if initialized:
                                    active_ticks.append({
                                        "block": block_number,
                                        "tick": tick,
                                        "liquidityGross": tick_data[0],
                                        "liquidityNet": tick_data[1],
                                    })
                            except Exception:
                                # Silent continue if a single tick fails
                                continue
            
            # print(f"Found {len(active_ticks)} active ticks at block {block_number}")
            return active_ticks
            
        except Exception as e:
            print(f"Error getting active ticks at block {block_number}: {e}")
            return []
    
    def generate_adaptive_block_samples(
        self, 
        start_block: int, 
        end_block: int, 
        base_step: int = 1000,
        threshold: float = 5.0
    ) -> List[int]:
        """
        Generate block samples with adaptive sampling based on volatility.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            base_step: Base number of blocks to skip between data points
            threshold: Price change percentage threshold for increasing sampling rate
            
        Returns:
            List of block numbers to sample
        """
        if not ENABLE_ADAPTIVE_SAMPLING:
            # If adaptive sampling is disabled, use regular sampling
            return list(range(start_block, end_block + 1, base_step))
        
        print("Using adaptive sampling based on price volatility")
        
        # Start with a coarse sampling to identify high volatility regions
        coarse_step = base_step * 5
        coarse_blocks = list(range(start_block, end_block + 1, coarse_step))
        
        # Collect data for coarse blocks
        coarse_data = []
        for block in tqdm(coarse_blocks, desc="Collecting coarse data for adaptive sampling"):
            data_point = self.get_pool_data_at_block(block)
            if data_point:
                coarse_data.append(data_point)
        
        # Calculate price changes between coarse blocks
        high_volatility_regions = []
        for i in range(1, len(coarse_data)):
            current = coarse_data[i]
            previous = coarse_data[i-1]
            
            # Calculate price change percentage
            price_change_pct = abs((current["price"] - previous["price"]) / previous["price"] * 100)
            
            # If price change exceeds threshold, mark this as a high volatility region
            if price_change_pct > threshold:
                region_start = previous["block"]
                region_end = current["block"]
                high_volatility_regions.append((region_start, region_end))
                print(f"High volatility region detected: blocks {region_start}-{region_end} "
                      f"with {price_change_pct:.2f}% price change")
        
        # Generate final block samples
        final_blocks = set()
        
        # Add regular samples
        for block in range(start_block, end_block + 1, base_step):
            final_blocks.add(block)
        
        # Add more frequent samples in high volatility regions
        for region_start, region_end in high_volatility_regions:
            # Use a smaller step in high volatility regions
            dense_step = max(base_step // 5, 1)
            for block in range(region_start, region_end + 1, dense_step):
                final_blocks.add(block)
        
        # Sort blocks and return
        return sorted(list(final_blocks))
    
    def process_blocks_parallel(
        self, 
        blocks: List[int], 
        include_active_ticks: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process blocks in parallel using ThreadPoolExecutor."""
        pool_data = []
        active_ticks_data = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a partial function with fixed parameters
            process_func = partial(self.get_pool_data_at_block, with_retries=True)
            
            # Submit all block processing tasks
            future_to_block = {executor.submit(process_func, block): block for block in blocks}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_block), 
                              total=len(blocks), 
                              desc="Processing blocks in parallel"):
                block = future_to_block[future]
                try:
                    data_point = future.result()
                    if data_point:
                        pool_data.append(data_point)
                        
                        # If we need tick data, process it separately
                        if include_active_ticks:
                            try:
                                # Add retry logic for getting active ticks with progressive fallback
                                max_retries = 3
                                retries = 0
                                # Start with default tick range and reduce if needed
                                tick_ranges = [30, 20, 10]  # Progressive fallback to smaller ranges
                                
                                success = False
                                while retries < max_retries and not success:
                                    try:
                                        # Use smaller tick range on each retry
                                        tick_range = tick_ranges[min(retries, len(tick_ranges)-1)]
                                        ticks = self.get_active_ticks_at_block(block, tick_range=tick_range)
                                        if ticks:  # Only extend if we got valid data
                                            active_ticks_data.extend(ticks)
                                            success = True
                                        else:
                                            # Empty result but no error, consider it a success
                                            success = True
                                    except Exception as inner_e:
                                        retries += 1
                                        if "gas limit" in str(inner_e).lower():
                                            print(f"Gas limit error for block {block}. Retrying with smaller tick range: {tick_ranges[min(retries, len(tick_ranges)-1)]}")
                                        else:
                                            print(f"Error getting ticks at block {block} (attempt {retries}/{max_retries}): {inner_e}")
                                        
                                        if retries >= max_retries:
                                            print(f"Failed to get active ticks at block {block} after {max_retries} attempts")
                                            break
                                        
                                        # Exponential backoff
                                        time.sleep(1 * retries)
                            except Exception as e:
                                print(f"Error in tick collection process for block {block}: {e}")
                except Exception as e:
                    print(f"Error processing block {block}: {e}")
        
        return pool_data, active_ticks_data
    
    def collect_historical_data(
        self, 
        start_block: int, 
        end_block: int, 
        block_step: int = None,
        include_active_ticks: bool = False,
        fetch_volume: bool = False
    ) -> Dict[str, Any]:
        """
        Collect historical data for a range of blocks with optimizations.
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            block_step: Number of blocks to skip between data points (uses DEFAULT_BLOCK_STEP if None)
            include_active_ticks: Whether to collect active ticks data (slower)
            fetch_volume: Whether to fetch trading volume data (requires external API)
            
        Returns:
            Dictionary with pool metadata and historical data
        """
        # Validate input
        if start_block >= end_block:
            raise ValueError("start_block must be less than end_block")
        
        # Use default block step if not specified
        if block_step is None:
            block_step = DEFAULT_BLOCK_STEP
        
        # Generate block numbers to collect data for
        if ENABLE_ADAPTIVE_SAMPLING:
            blocks = self.generate_adaptive_block_samples(
                start_block, 
                end_block, 
                base_step=block_step,
                threshold=ADAPTIVE_SAMPLING_THRESHOLD
            )
        else:
            blocks = list(range(start_block, end_block + 1, block_step))
        
        print(f"Collecting data for {len(blocks)} blocks from {start_block} to {end_block}")
        
        # Collect data with progress bar
        pool_data = []
        active_ticks_data = []
        
        if ENABLE_PARALLEL_PROCESSING:
            # Process blocks in parallel
            print(f"Using parallel processing with {MAX_WORKERS} workers")
            
            # Split blocks into batches for better progress tracking
            batches = [blocks[i:i + BATCH_SIZE] for i in range(0, len(blocks), BATCH_SIZE)]
            
            for batch in tqdm(batches, desc="Processing batches"):
                batch_pool_data, batch_ticks_data = self.process_blocks_parallel(
                    batch, include_active_ticks
                )
                pool_data.extend(batch_pool_data)
                active_ticks_data.extend(batch_ticks_data)
        else:
            # Process blocks sequentially
            for block in tqdm(blocks, desc="Collecting pool data"):
                # Get pool data at this block
                data_point = self.get_pool_data_at_block(block)
                if data_point:
                    pool_data.append(data_point)
                    
                    # Optionally collect active ticks data
                    if include_active_ticks:
                        ticks = self.get_active_ticks_at_block(block)
                        active_ticks_data.extend(ticks)
        
        # Calculate additional metrics based on collected data
        self.calculate_derived_metrics(pool_data)
        
        # Fetch trading volume data if requested
        if fetch_volume:
            volume_data = self.fetch_trading_volume(pool_data)
            
            # Add volume data to pool data
            if volume_data:
                self.merge_volume_data(pool_data, volume_data)
        else:
            volume_data = []
        
        # Build result
        result = {
            "metadata": {
                "pool_address": self.pool_address,
                "token0": {
                    "address": self.token0_address,
                    "symbol": self.token0_symbol,
                    "decimals": self.token0_decimals
                },
                "token1": {
                    "address": self.token1_address,
                    "symbol": self.token1_symbol,
                    "decimals": self.token1_decimals
                },
                "fee": self.fee,
                "tick_spacing": self.tick_spacing,
                "start_block": start_block,
                "end_block": end_block,
                "block_step": block_step,
                "adaptive_sampling": ENABLE_ADAPTIVE_SAMPLING,
                "parallel_processing": ENABLE_PARALLEL_PROCESSING,
                "data_points": len(pool_data),
                "collection_timestamp": int(time.time()),
                "collection_datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "volume_data_available": len(volume_data) > 0
            },
            "pool_data": pool_data
        }
        
        if include_active_ticks:
            result["active_ticks_data"] = active_ticks_data
            
        if volume_data:
            result["volume_data"] = volume_data
            
        return result
    
    def calculate_derived_metrics(self, pool_data: List[Dict[str, Any]]) -> None:
        """Calculate derived metrics for analysis."""
        if not pool_data or len(pool_data) < 2:
            return
        
        # Sort by block number to ensure proper order
        pool_data.sort(key=lambda x: x["block"])
        
        for i in range(1, len(pool_data)):
            current = pool_data[i]
            previous = pool_data[i-1]
            
            # Calculate price change
            if "price" in current and "price" in previous and previous["price"] != 0:
                current["price_change"] = current["price"] - previous["price"]
                current["price_change_pct"] = (current["price"] - previous["price"]) / previous["price"] * 100
            
            # Calculate liquidity change
            if "liquidity" in current and "liquidity" in previous:
                current["liquidity_change"] = current["liquidity"] - previous["liquidity"]
                if previous["liquidity"] > 0:
                    current["liquidity_change_pct"] = (current["liquidity"] - previous["liquidity"]) / previous["liquidity"] * 100
                else:
                    # Handle zero liquidity case
                    if current["liquidity"] > 0:
                        # If current liquidity is positive but previous was zero, it's a 100% increase from 0
                        current["liquidity_change_pct"] = 100.0
                    else:
                        # Both zero, no change
                        current["liquidity_change_pct"] = 0.0
            
            # Calculate time difference
            if "timestamp" in current and "timestamp" in previous:
                time_diff = current["timestamp"] - previous["timestamp"]
                current["time_diff_seconds"] = time_diff
                
                # Calculate approximate annualized volatility based on price changes
                if "price_change_pct" in current and time_diff > 0:
                    # Annualized volatility approximation
                    volatility_daily = abs(current["price_change_pct"]) * (86400 / time_diff) ** 0.5
                    current["est_volatility_daily"] = volatility_daily
    
    def save_data_to_csv(self, data: Dict[str, Any], output_file: str) -> None:
        """Save collected data to CSV files."""
        # Create base filename without extension
        
        # Save pool data
        if "pool_data" in data and data["pool_data"]:
            df_pool = pd.DataFrame(data["pool_data"])
            df_pool.to_csv(f"{output_file}_pool_data.csv", index=False)
            print(f"Pool data saved to {output_file}_pool_data.csv")
        
        # Save active ticks data if available
        if "active_ticks_data" in data and data["active_ticks_data"]:
            df_ticks = pd.DataFrame(data["active_ticks_data"])
            df_ticks.to_csv(f"{output_file}_ticks_data.csv", index=False)
            print(f"Ticks data saved to {output_file}_ticks_data.csv")
        
        # Save metadata
        if "metadata" in data:
            with open(f"{output_file}_metadata.json", 'w') as f:
                json.dump(data["metadata"], f, indent=2)
            print(f"Metadata saved to {output_file}_metadata.json")
    
    def plot_price_liquidity(self, data: Dict[str, Any], output_file: str = None) -> None:
        """Plot price and liquidity over time."""
        if "pool_data" not in data or not data["pool_data"]:
            print("No data to plot")
            return
        
        df = pd.DataFrame(data["pool_data"])
        
        # Convert timestamp to datetime for better plotting
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Create subplots
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot price on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'Price ({self.token1_symbol}/{self.token0_symbol})', color=color)
        ax1.plot(df['datetime'], df['price'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot liquidity on right axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Liquidity', color=color)
        ax2.plot(df['datetime'], df['liquidity'], color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add title
        fig.suptitle(f'{self.token0_symbol}/{self.token1_symbol} {self.fee/10000}% Pool - Price and Liquidity', fontsize=16)
        fig.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        else:
            plt.show()
    
    def calculate_impermanent_loss(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate impermanent loss for the pool over the time period.
        
        Impermanent Loss (IL) is the difference in value between holding assets in a 
        liquidity pool versus holding them in a wallet.
        """
        if "pool_data" not in data or len(data["pool_data"]) < 2:
            print("Insufficient data to calculate impermanent loss")
            return None
        
        df = pd.DataFrame(data["pool_data"])
        
        # Sort by block number
        df = df.sort_values("block")
        
        # Use the first price as the reference price
        initial_price = df.iloc[0]["price"]
        
        # Calculate price ratio for each data point
        df["price_ratio"] = df["price"] / initial_price
        
        # Calculate impermanent loss percentage using the formula:
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        df["impermanent_loss_pct"] = 2 * np.sqrt(df["price_ratio"]) / (1 + df["price_ratio"]) - 1
        
        # Convert to percentage
        df["impermanent_loss_pct"] = df["impermanent_loss_pct"] * 100
        
        return df[["block", "timestamp", "datetime", "price", "price_ratio", "impermanent_loss_pct"]]
    
    def analyze_tick_concentration(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Analyze liquidity concentration around active ticks."""
        if "active_ticks_data" not in data or not data["active_ticks_data"]:
            print("No tick data available for analysis")
            return None
        
        df_ticks = pd.DataFrame(data["active_ticks_data"])
        
        # Group by block and calculate liquidity distribution
        blocks = df_ticks["block"].unique()
        results = []
        
        for block in blocks:
            block_ticks = df_ticks[df_ticks["block"] == block]
            
            # Calculate total liquidity
            total_liquidity = block_ticks["liquidityGross"].sum()
            
            # Find tick with maximum liquidity
            max_liquidity_tick = block_ticks.loc[block_ticks["liquidityGross"].idxmax()]
            
            # Calculate concentration (% of liquidity in the top 3 ticks)
            top_ticks = block_ticks.nlargest(3, "liquidityGross")
            top_ticks_liquidity = top_ticks["liquidityGross"].sum()
            concentration = (top_ticks_liquidity / total_liquidity) * 100 if total_liquidity > 0 else 0
            
            results.append({
                "block": block,
                "total_liquidity": total_liquidity,
                "max_liquidity_tick": max_liquidity_tick["tick"],
                "max_liquidity_value": max_liquidity_tick["liquidityGross"],
                "top_3_ticks_concentration_pct": concentration
            })
        
        return pd.DataFrame(results)
    
    def fetch_trading_volume(self, pool_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch trading volume data for the pool.
        
        This uses the public Uniswap Subgraph API to get volume data.
        Includes retry logic and error handling for API failures.
        
        Returns:
            List of dictionaries with timestamp and volume data
        """
        if not pool_data:
            return []
            
        # Get date range from pool data
        start_timestamp = min(p["timestamp"] for p in pool_data)
        end_timestamp = max(p["timestamp"] for p in pool_data)
        
        # Convert to datetime for formatting
        start_date = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d')
        end_date = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d')
        
        print(f"Fetching volume data from {start_date} to {end_date}")
        
        # Get API key from environment variables
        api_key = os.getenv("THEGRAPH_API_KEY")
        if not api_key:
            print("Warning: THEGRAPH_API_KEY not found in environment variables")
        
        # Updated subgraph endpoint using the gateway URL and subgraph ID from the screenshot
        subgraph_endpoint = "https://gateway.thegraph.com/api"
        subgraph_id = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
        
        # List of subgraph endpoints to try (primary and backups)
        subgraph_endpoints = [
            f"{subgraph_endpoint}/{api_key}/subgraphs/id/{subgraph_id}"
        ]
        
        # Build GraphQL query
        query = f"""
        {{
          poolDayDatas(
            where: {{ 
              pool: "{self.pool_address.lower()}"
              date_gte: {start_timestamp}
              date_lte: {end_timestamp}
            }}
            orderBy: date
            orderDirection: asc
          ) {{
            date
            volumeUSD
            tvlUSD
            volumeToken0
            volumeToken1
            feesUSD
          }}
        }}
        """
        
        # Try each endpoint with retries
        max_retries = 3
        for endpoint in subgraph_endpoints:
            retries = 0
            while retries < max_retries:
                try:
                    print(f"Trying subgraph endpoint: {endpoint} (attempt {retries+1}/{max_retries})")
                    
                    # Execute query with timeout
                    response = requests.post(
                        endpoint,
                        json={"query": query},
                        timeout=30  # 30 second timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if "data" in data and "poolDayDatas" in data["data"]:
                            # Extract and format volume data
                            volume_data = []
                            for day_data in data["data"]["poolDayDatas"]:
                                try:
                                    volume_data.append({
                                        "timestamp": int(day_data["date"]),
                                        "datetime": datetime.fromtimestamp(int(day_data["date"])).strftime('%Y-%m-%d'),
                                        "volume_usd": float(day_data["volumeUSD"]),
                                        "tvl_usd": float(day_data["tvlUSD"]),
                                        "volume_token0": float(day_data["volumeToken0"]),
                                        "volume_token1": float(day_data["volumeToken1"]),
                                        "fees_usd": float(day_data["feesUSD"])
                                    })
                                except (ValueError, KeyError, TypeError) as e:
                                    print(f"Error processing day data: {e}")
                                    continue
                            
                            print(f"Successfully fetched {len(volume_data)} days of volume data from {endpoint}")
                            return volume_data
                        else:
                            print(f"No volume data returned from API endpoint {endpoint}")
                    else:
                        print(f"Error {response.status_code} from API endpoint {endpoint}: {response.text[:200]}")
                    
                except (requests.RequestException, requests.Timeout) as e:
                    print(f"Network error with API endpoint {endpoint}: {str(e)}")
                
                # Increment retry counter and add exponential backoff
                retries += 1
                if retries < max_retries:
                    wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8... seconds
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # If we reach here, all endpoints failed
        print("All subgraph endpoints failed, unable to fetch volume data")
        return []
    
    def merge_volume_data(self, pool_data: List[Dict[str, Any]], volume_data: List[Dict[str, Any]]) -> None:
        """
        Merge volume data into pool data by matching timestamps.
        
        Args:
            pool_data: List of pool data points
            volume_data: List of volume data points
        """
        if not volume_data:
            return
            
        # Create mapping of dates to volume data
        volume_by_date = {}
        for v in volume_data:
            dt = datetime.fromtimestamp(v["timestamp"]).strftime('%Y-%m-%d')
            volume_by_date[dt] = v
        
        # Go through pool data and add volume info
        for p in pool_data:
            dt = datetime.fromtimestamp(p["timestamp"]).strftime('%Y-%m-%d')
            if dt in volume_by_date:
                v = volume_by_date[dt]
                p["daily_volume_usd"] = v["volume_usd"]
                p["daily_volume_token0"] = v["volume_token0"]
                p["daily_volume_token1"] = v["volume_token1"]
                p["tvl_usd"] = v["tvl_usd"]
                p["fees_usd"] = v["fees_usd"]

    # Dynamically determine tick range based on pool characteristics
    def determine_optimal_tick_range(self, pool_data_history):
        if not pool_data_history or len(pool_data_history) < 10:
            return 30  # Default
        
        # Calculate tick volatility from recent history
        recent_ticks = [data["tick"] for data in pool_data_history[-10:]]
        tick_range = max(10, int(np.std(recent_ticks) * 3))  # Cover 3 standard deviations
        return min(20, tick_range)  # Cap at 20 ticks

    def cleanup_multicall_cache(self, max_size=1000):
        """Clean up the multicall cache when it gets too large."""
        if len(self.multicall_cache) > max_size:
            # Remove oldest 20% of entries
            remove_count = max_size // 5
            keys_to_remove = list(self.multicall_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.multicall_cache[key]
            print(f"Cleaned up multicall cache, removed {remove_count} entries")
    
    def execute_multicall_with_cache(self, calls, block_id, call_type="general"):
        """Execute multicall with caching to reduce RPC calls."""
        import time
        
        # Create a cache key based on the calls and block_id
        cache_key = f"{call_type}_{block_id}_{hash(str(calls))}"
        
        # Check if we need to clean up the cache
        if len(self.multicall_cache) > self.max_cache_size:
            self.cleanup_multicall_cache(self.max_cache_size)
        
        # Return cached result if available
        if cache_key in self.multicall_cache:
            return self.multicall_cache[cache_key]
        
        # Execute the multicall and track performance
        start_time = time.time()
        success = False
        batch_size = len(calls)
        
        try:
            # Use the RPC manager to execute the multicall
            result = self.rpc_manager.execute_multicall(calls, block_id)
            
            # Validate the result
            if result is None:
                print(f"Warning: Multicall returned None for {call_type}")
                result = {}
            elif not isinstance(result, dict):
                print(f"Warning: Multicall returned non-dict type: {type(result)}")
                result = {}
            
            # Cache the result
            self.multicall_cache[cache_key] = result
            success = True
            return result
        except Exception as e:
            # Log the error and re-raise
            print(f"Multicall failed for {call_type} with batch size {batch_size}: {e}")
            raise
        finally:
            # Track performance regardless of success or failure
            execution_time = time.time() - start_time
            self.track_multicall_performance(call_type, batch_size, success, execution_time)

    def update_batch_size_history(self, pool_address, call_type, success, batch_size):
        """Update batch size history based on success/failure of multicalls."""
        # If successful with current batch size, try slightly larger next time
        if success and call_type == "ticks":
            # Be more conservative with increasing batch size
            current_size = self.tick_batch_size_history.get(pool_address, 10)
            # Only increase if we've had multiple successes at this size
            if batch_size == current_size:
                # Increase by 1 instead of 2 to be more gradual
                self.tick_batch_size_history[pool_address] = min(15, batch_size + 1)
            else:
                # If we're using a different size than what's in history, just update to current
                self.tick_batch_size_history[pool_address] = batch_size
        # If failed, reduce batch size more aggressively for next time
        elif not success and call_type == "ticks":
            # Reduce more aggressively on failure
            self.tick_batch_size_history[pool_address] = max(3, batch_size // 2)

    # Get metadata for multiple blocks in one request
    def get_metadata_for_blocks(self, blocks):
        # Make a single multicall for metadata that applies to all blocks
        calls = [
            Call(self.token0_address, ['decimals()(uint8)'], [['token0_decimals', None]]),
            Call(self.token1_address, ['decimals()(uint8)'], [['token1_decimals', None]]),
            # Add other metadata calls
        ]
        return self.rpc_manager.execute_multicall(calls)

    # Group blocks that are close to each other
    def group_blocks_by_proximity(self, blocks, max_distance=100):
        blocks = sorted(blocks)
        groups = []
        current_group = [blocks[0]]
        
        for i in range(1, len(blocks)):
            if blocks[i] - blocks[i-1] <= max_distance:
                current_group.append(blocks[i])
            else:
                groups.append(current_group)
                current_group = [blocks[i]]
        
        if current_group:
            groups.append(current_group)
        
        return groups

    # Process groups in parallel instead of individual blocks
    def process_block_group(self, block_group):
        """Process a group of blocks with batched multicall requests."""
        results = []
        
        # Get common metadata for all blocks in one call
        try:
            metadata_calls = [
                Call(
                    self.token0_address,
                    ['decimals()(uint8)'],
                    [['token0_decimals', None]]
                ),
                Call(
                    self.token1_address,
                    ['decimals()(uint8)'],
                    [['token1_decimals', None]]
                )
            ]
            metadata = self.execute_multicall_with_cache(metadata_calls, None, "metadata")
        except Exception as e:
            print(f"Error getting metadata: {e}")
            metadata = {}
        
        # Prepare batch calls for all blocks
        batch_calls = []
        for block in block_group:
            # Add slot0 call for each block
            batch_calls.append(
                Call(
                    self.pool_address,
                    ['slot0()(uint160,int24,uint16,uint16,uint16,uint8,bool)'],
                    [['sqrtPriceX96', None], ['tick', None], ['observationIndex', None],
                     ['observationCardinality', None], ['observationCardinalityNext', None],
                     ['feeProtocol', None], ['unlocked', None]],
                    block_id=block
                )
            )
            
            # Add liquidity call for each block
            batch_calls.append(
                Call(
                    self.pool_address,
                    ['liquidity()(uint128)'],
                    [['liquidity', None]],
                    block_id=block
                )
            )
        
        # Execute batch calls in chunks to avoid gas limit issues
        CHUNK_SIZE = self.get_optimal_batch_size(self.pool_address, "pool_data")
        for i in range(0, len(batch_calls), CHUNK_SIZE):
            chunk = batch_calls[i:i+CHUNK_SIZE]
            try:
                # Use cached multicall with RPC manager
                chunk_results = self.execute_multicall_with_cache(chunk, None, "pool_data")
                
                # Process results and map them back to blocks
                # This is a simplified example - you'll need to implement the mapping logic
                for j in range(0, len(chunk), 2):  # Process in pairs (slot0 + liquidity)
                    if j+1 < len(chunk):
                        block_id = chunk[j].block_id
                        
                        # Extract data from results
                        slot0_key = f"slot0()(uint160,int24,uint16,uint16,uint16,uint8,bool)"
                        liquidity_key = f"liquidity()(uint128)"
                        
                        # Add processed result
                        if slot0_key in chunk_results and liquidity_key in chunk_results:
                            timestamp = self.get_block_timestamp(block_id)
                            
                            # Create result object
                            block_result = {
                                "block": block_id,
                                "timestamp": timestamp,
                                "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                # Add other fields from chunk_results
                            }
                            results.append(block_result)
            
            except Exception as e:
                print(f"Error processing batch chunk: {e}")
                # Fall back to individual processing if needed
                for j in range(i, min(i+CHUNK_SIZE, len(batch_calls)), 2):
                    if j+1 < len(batch_calls):
                        try:
                            block_id = batch_calls[j].block_id
                            # Use direct RPC manager calls as fallback
                            slot0 = self.rpc_manager.call_contract_function(
                                self.pool_contract.functions.slot0(),
                                block_identifier=block_id
                            )
                            liquidity = self.rpc_manager.call_contract_function(
                                self.pool_contract.functions.liquidity(),
                                block_identifier=block_id
                            )
                            
                            # Process results
                            timestamp = self.get_block_timestamp(block_id)
                            # Create result object
                            block_result = {
                                "block": block_id,
                                "timestamp": timestamp,
                                "datetime": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                                # Add other fields from direct calls
                            }
                            results.append(block_result)
                        except Exception as inner_e:
                            print(f"Error processing individual block {block_id}: {inner_e}")
        
        return results

    def process_block_groups_parallel(self, block_groups):
        """Process multiple block groups in parallel using ThreadPoolExecutor."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_group = {executor.submit(self.process_block_group, group): group for group in block_groups}
            for future in concurrent.futures.as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    group_results = future.result()
                    if group_results:
                        results.extend(group_results)
                except Exception as e:
                    print(f"Error processing block group {group[0]}-{group[-1]}: {e}")
        
        return results


def get_blocks_for_time_range(web3_manager: RPCManager, days: int) -> Tuple[int, int]:
    """Get start and end block numbers for a time range in days."""
    # Get current block
    end_block = web3_manager.web3.eth.block_number
    
    # Estimate start block based on average block time
    blocks_to_go_back = days * BLOCKS_PER_DAY
    start_block = max(1, end_block - blocks_to_go_back)
    
    return start_block, end_block


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Sharpe Ratio = (Mean Portfolio Return - Risk Free Rate) / Portfolio Standard Deviation
    """
    if not returns:
        return None
    
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_dev = np.std(returns_array)
    
    if std_dev == 0:
        return None
    
    sharpe = (mean_return - risk_free_rate) / std_dev
    return sharpe


def analyze_backtest_strategy(
    data: Dict[str, Any],
    rebalance_threshold: float = 5.0,  # Rebalance when price changes by 5%
    tick_range: int = 10  # Number of ticks on each side to provide liquidity
) -> Dict[str, Any]:
    """
    Backtest a simple dynamic liquidity provision strategy.
    
    Strategy:
    1. Provide liquidity in a range around the current price (±tick_range)
    2. Rebalance when price moves beyond threshold percentage
    """
    if "pool_data" not in data or len(data["pool_data"]) < 2:
        print("Insufficient data for strategy backtesting")
        return None
    
    df = pd.DataFrame(data["pool_data"])
    df = df.sort_values("block")
    
    # Initialize strategy variables
    initial_price = df.iloc[0]["price"]
    current_range_low = df.iloc[0]["tick"] - tick_range
    current_range_high = df.iloc[0]["tick"] + tick_range
    last_rebalance_price = initial_price
    
    rebalance_events = []
    daily_returns = []
    fees_collected = []
    
    # Track positions and rebalancing
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        
        current_price = current["price"]
        price_change_pct = abs((current_price - last_rebalance_price) / last_rebalance_price * 100)
        
        # Check if we need to rebalance
        if price_change_pct > rebalance_threshold:
            # Record rebalance event
            rebalance_events.append({
                "block": current["block"],
                "timestamp": current["timestamp"],
                "old_range_low": current_range_low,
                "old_range_high": current_range_high,
                "new_range_low": current["tick"] - tick_range,
                "new_range_high": current["tick"] + tick_range,
                "price": current_price,
                "price_change_pct": price_change_pct
            })
            
            # Update range and last rebalance price
            current_range_low = current["tick"] - tick_range
            current_range_high = current["tick"] + tick_range
            last_rebalance_price = current_price
            
            # Simulate fee collection and IL
            if "time_diff_seconds" in current:
                # Simple fee estimation based on time passed and volatility
                time_fraction = current["time_diff_seconds"] / (24 * 60 * 60)  # Fraction of a day
                if "est_volatility_daily" in current:
                    # Higher volatility typically means more trading and fees
                    estimated_daily_fee = 0.0001 * current["est_volatility_daily"] * float(current["liquidity"])
                    fee_collected = estimated_daily_fee * time_fraction
                    fees_collected.append(fee_collected)
                
                # Calculate IL for this period
                price_ratio = current_price / last_rebalance_price
                il_pct = (2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1) * 100
                
                # Estimate return: fees earned minus IL
                if fee_collected > 0:
                    period_return = fee_collected - abs(il_pct)
                    daily_returns.append(period_return)
    
    # Calculate strategy metrics
    avg_daily_return = np.mean(daily_returns) if daily_returns else 0
    sharpe = calculate_sharpe_ratio(daily_returns) if daily_returns else 0
    total_fees = sum(fees_collected)
    
    return {
        "rebalance_events": rebalance_events,
        "rebalance_count": len(rebalance_events),
        "avg_daily_return": avg_daily_return,
        "estimated_sharpe": sharpe,
        "total_fees_collected": total_fees
    }


def main():
    
    # Override global settings with command line arguments
    global ENABLE_ADAPTIVE_SAMPLING, ENABLE_PARALLEL_PROCESSING, MAX_WORKERS, BATCH_SIZE
    
    """Main function to run the data collection."""
    parser = argparse.ArgumentParser(description="Collect Uniswap V3 pool historical data")
    parser.add_argument("--pool", type=str, default=DEFAULT_POOL_ADDRESS,
                        help="Uniswap V3 pool address")
    parser.add_argument("--days", type=int, default=365,
                        help="Number of days to collect data for (default: 365)")
    parser.add_argument("--step", type=int, default=DEFAULT_BLOCK_STEP,
                        help=f"Block step for data collection (default: {DEFAULT_BLOCK_STEP})")
    parser.add_argument("--output", type=str, default="uniswap_v3_data",
                        help="Base name for output files (default: uniswap_v3_data)")
    parser.add_argument("--ticks", action="store_true",
                        help="Include active ticks data (slower)")
    parser.add_argument("--no-ticks", action="store_true",
                        help="Explicitly disable active ticks collection even if other options would enable it")
    parser.add_argument("--backtest", action="store_true",
                        help="Run strategy backtesting")
    parser.add_argument("--volume", action="store_true",
                        help="Fetch trading volume data for more accurate fee calculation")
    parser.add_argument("--export-ticks", action="store_true",
                        help="Export tick data in a format usable by the strategy script")
    parser.add_argument("--adaptive", action="store_true", dest="adaptive_sampling",
                        help="Enable adaptive sampling based on volatility")
    parser.add_argument("--no-adaptive", action="store_false", dest="adaptive_sampling",
                        help="Disable adaptive sampling")
    parser.add_argument("--parallel", action="store_true", dest="parallel_processing",
                        help="Enable parallel processing")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel_processing",
                        help="Disable parallel processing")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Number of worker threads for parallel processing (default: {MAX_WORKERS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for parallel processing (default: {BATCH_SIZE})")
    parser.add_argument("--tick-range", type=int, default=10,
                        help="Number of ticks to check on each side of the current tick (default: 10)")
    parser.add_argument("--multicall-batch-size", type=int, default=10,
                        help="Batch size for multicall operations (default: 10, reduce to 5 if gas limit errors occur)")
    
    # Set default values from environment variables
    parser.set_defaults(
        adaptive_sampling=ENABLE_ADAPTIVE_SAMPLING,
        parallel_processing=ENABLE_PARALLEL_PROCESSING
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Assign values to global variables
    ENABLE_ADAPTIVE_SAMPLING = args.adaptive_sampling
    ENABLE_PARALLEL_PROCESSING = args.parallel_processing
    MAX_WORKERS = args.workers
    BATCH_SIZE = args.batch_size
    
    # Handle the no-ticks override
    include_ticks = args.ticks and not args.no_ticks
    
    # Print configuration
    print("\n" + "="*60)
    print(f"\033[1;36mUniswap V3 Data Collector\033[0m")
    print("="*60)
    print(f"\033[1mConfiguration:\033[0m")
    print(f"  • Pool Address: \033[1;33m{args.pool}\033[0m")
    print(f"  • Time Period: \033[1;33m{args.days}\033[0m days")
    print(f"  • Block Step: \033[1;33m{args.step}\033[0m")
    print(f"  • Adaptive Sampling: \033[1;32mEnabled\033[0m" if ENABLE_ADAPTIVE_SAMPLING else f"  • Adaptive Sampling: \033[1;31mDisabled\033[0m")
    print(f"  • Parallel Processing: \033[1;32mEnabled\033[0m" if ENABLE_PARALLEL_PROCESSING else f"  • Parallel Processing: \033[1;31mDisabled\033[0m")
    if ENABLE_PARALLEL_PROCESSING:
        print(f"  • Worker Threads: \033[1;33m{MAX_WORKERS}\033[0m")
        print(f"  • Batch Size: \033[1;33m{BATCH_SIZE}\033[0m")
    print(f"  • Include Ticks: \033[1;32mYes\033[0m" if include_ticks else f"  • Include Ticks: \033[1;31mNo\033[0m")
    print(f"  • Fetch Volume: \033[1;32mYes\033[0m" if args.volume else f"  • Fetch Volume: \033[1;31mNo\033[0m")
    print(f"  • Run Backtesting: \033[1;32mYes\033[0m" if args.backtest else f"  • Run Backtesting: \033[1;31mNo\033[0m")
    print("-"*60)
    
    # Initialize collector
    collector = UniswapV3PoolDataCollector(args.pool)
    
    # Get block range for time period
    print("\n\033[1mDetermining block range...\033[0m")
    start_block, end_block = get_blocks_for_time_range(rpc_manager, args.days)
    
    print(f"  • Collecting data for approximately \033[1;33m{args.days}\033[0m days")
    print(f"  • Block range: \033[1;33m{start_block}\033[0m to \033[1;33m{end_block}\033[0m")
    
    # Collect data
    print("\n\033[1mStarting data collection...\033[0m")
    
    # Create a loading animation
    def loading_animation(stop_event):
        animation = "|/-\\"
        idx = 0
        while not stop_event.is_set():
            print(f"\r  • Processing blocks... {animation[idx % len(animation)]}", end="")
            idx += 1
            time.sleep(0.1)
    
    import threading
    import time
    stop_loading = threading.Event()
    loading_thread = threading.Thread(target=loading_animation, args=(stop_loading,))
    loading_thread.daemon = True
    loading_thread.start()
    
    try:
        data = collector.collect_historical_data(
            start_block=start_block,
            end_block=end_block,
            block_step=args.step,
            include_active_ticks=include_ticks,
            fetch_volume=args.volume
        )
    finally:
        stop_loading.set()
        loading_thread.join(timeout=1)
    
    print(f"\r  • Processing blocks... \033[1;32mComplete!\033[0m" + " "*20)
    
    # Save data
    print("\n\033[1mSaving collected data...\033[0m")
    output_file = f"{args.output}_{collector.fee/10000}"
    collector.save_data_to_csv(data, output_file)
    print(f"  • Data saved to \033[1;36m{output_file}.csv\033[0m")
    
    # Generate plots
    print("\n\033[1mGenerating visualizations...\033[0m")
    collector.plot_price_liquidity(data, f"{output_file}_plot.png")
    print(f"  • Price/liquidity plot saved to \033[1;36m{output_file}_plot.png\033[0m")
    
    # Calculate impermanent loss
    print("\n\033[1mCalculating impermanent loss...\033[0m")
    il_data = collector.calculate_impermanent_loss(data)
    if il_data is not None:
        il_data.to_csv(f"{output_file}_impermanent_loss.csv", index=False)
        print(f"  • Impermanent loss data saved to \033[1;36m{output_file}_impermanent_loss.csv\033[0m")
    
    # Export tick data in a format usable by strategy script if requested
    if args.export_ticks and "active_ticks_data" in data:
        print("\n\033[1mExporting tick data...\033[0m")
        tick_data_by_block = {}
        for tick_data in data["active_ticks_data"]:
            block = str(tick_data["block"])
            tick = tick_data["tick"]
            liquidity = tick_data["liquidityGross"]
            
            if block not in tick_data_by_block:
                tick_data_by_block[block] = {}
            
            tick_data_by_block[block][tick] = float(liquidity)
        
        # Save to JSON
        with open(f"{output_file}_tick_data.json", 'w') as f:
            json.dump(tick_data_by_block, f)
        
        print(f"  • Tick data exported to \033[1;36m{output_file}_tick_data.json\033[0m")
    
    # Run strategy backtesting if requested
    if args.backtest:
        print("\n\033[1mRunning strategy backtesting...\033[0m")
        strategy_results = analyze_backtest_strategy(data)
        if strategy_results:
            # Save rebalance events
            rebalance_df = pd.DataFrame(strategy_results["rebalance_events"])
            rebalance_df.to_csv(f"{output_file}_rebalance_events.csv", index=False)
            print(f"  • Rebalance events saved to \033[1;36m{output_file}_rebalance_events.csv\033[0m")
            
            # Print summary metrics
            print("\n\033[1;32mStrategy Backtesting Results:\033[0m")
            print(f"  • Total Rebalances: \033[1;33m{strategy_results['rebalance_count']}\033[0m")
            print(f"  • Avg Daily Return: \033[1;33m{strategy_results['avg_daily_return']:.4f}%\033[0m")
            print(f"  • Estimated Sharpe Ratio: \033[1;33m{strategy_results['estimated_sharpe']:.4f}\033[0m")
            print(f"  • Total Fees Collected: \033[1;33m{strategy_results['total_fees_collected']:.6f}\033[0m")
    
    print("\n" + "="*60)
    print(f"\033[1;32mData collection complete!\033[0m")
    print(f"  • \033[1;33m{len(data['pool_data'])}\033[0m data points collected")
    
    # Print info about additional data if available
    if "volume_data" in data:
        print(f"  • Volume data collected: \033[1;33m{len(data['volume_data'])}\033[0m days")
    if "active_ticks_data" in data:
        print(f"  • Active ticks data collected: \033[1;33m{len(data['active_ticks_data'])}\033[0m ticks")
    
    # Print performance statistics
    if ENABLE_ADAPTIVE_SAMPLING:
        print("\n\033[1mAdaptive Sampling Statistics:\033[0m")
        if "metadata" in data and "block_step" in data["metadata"]:
            base_count = (end_block - start_block) // data["metadata"]["block_step"] + 1
            print(f"  • Collected \033[1;33m{len(data['pool_data'])}\033[0m data points with adaptive sampling")
            print(f"  • Regular sampling would have collected approximately \033[1;33m{base_count}\033[0m data points")
            if len(data['pool_data']) > base_count:
                print(f"  • Added \033[1;32m{len(data['pool_data']) - base_count}\033[0m extra data points in high volatility regions")
    
    if ENABLE_PARALLEL_PROCESSING:
        print("\n\033[1mParallel Processing Statistics:\033[0m")
        print(f"  • Used \033[1;33m{MAX_WORKERS}\033[0m worker threads with batch size \033[1;33m{BATCH_SIZE}\033[0m")
    
    print("="*60)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease report this issue with the error details above.") 