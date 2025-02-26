# Uniswap V3 LP Toolkit – Simulate, Backtest & Optimize Like a Degen

<div align="center">
  <img src="img/uni.jpeg" alt="Uniswap V3 LP Toolkit" width="100%"/>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![DeFi](https://img.shields.io/badge/DeFi-Uniswap_V3-blue)
[![Web3](https://img.shields.io/badge/Web3-6.0.0-orange.svg)](https://github.com/ethereum/web3.py)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-151F6D.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.0-11557c.svg)](https://matplotlib.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.0.0-3F4F75.svg)](https://plotly.com/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-77AAAD.svg)](https://seaborn.pydata.org/)
[![tqdm](https://img.shields.io/badge/tqdm-4.67.1-green.svg)](https://github.com/tqdm/tqdm)
[![Ethereum](https://img.shields.io/badge/Ethereum-ETH-3C3C3D.svg)](https://ethereum.org/)
[![Backtesting](https://img.shields.io/badge/Backtesting-Strategy-brightgreen.svg)](https://github.com/yourusername/uniswap-v3-pool-data)
[![Multicall](https://img.shields.io/badge/Multicall-Optimized-blue.svg)](https://github.com/mds1/multicall)

## 👨‍💻 Core Contributor

[![Author](https://img.shields.io/badge/Author-ilia.eth-lightgrey?style=for-the-badge)](https://github.com/yourusername)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/ilia_0x)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](http://twitter.com/ilia_0x)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maksimenka/)
[![Ethereum](https://img.shields.io/badge/Donate_ETH-3C3C3D?style=for-the-badge&logo=ethereum&logoColor=white)](https://etherscan.io/address/0x8746B03bc7297359818C30e378AF507379E79a07) 

## 📝 Project Description

A sophisticated toolkit for collecting Uniswap V3 pool data and backtesting dynamic liquidity provision strategies with realistic trading costs, impermanent loss, and comprehensive performance metrics.

_____________
<div align="center">

### 🚨 <span style="color:red"> WARNING </span> 🚨
<div style="color:red">
  
**This code is deep in the trenches of development—not production-ready.**  
**If you're not a Python chad or a true DeFi degen, stay away before you get rekt.**  
**Touch at your own risk. 🔥🐸💀**
  
</div>
</div>

_________________


## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Strategy Backtesting](#strategy-backtesting)
- [Strategy Types](#strategy-types)
- [Performance Metrics](#performance-metrics)
- [Output Files](#output-files)
- [Analyzing Results](#analyzing-results)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project provides tools to analyze Uniswap V3 pools and develop optimized liquidity provision strategies. It consists of two main components:

1. **Pool Data Collector** (`pool_data.py`): Collects comprehensive historical data from Uniswap V3 pools, including trading volume and tick-specific liquidity.
2. **Liquidity Strategy** (`liquidity_strategy.py`): Implements and backtests various liquidity provision strategies with realistic modeling of fees, slippage, and transaction costs.

The toolkit helps liquidity providers optimize their capital efficiency by determining when to rebalance positions and how to size position ranges based on historical volatility, with full portfolio value tracking and advanced performance metrics.

## ✨ Features

### Pool Data Collector
- **Robust RPC Management**:
  - Automatic failover between multiple RPC providers
  - Smart rate limit detection and cooldown periods
  - Exponential backoff retry mechanism
  - Prioritization of reliable RPC endpoints
- **Optimized Data Collection**:
  - Multicall batching with dynamic batch sizing
  - Result caching to reduce redundant RPC calls
  - Adaptive sampling based on price volatility
  - Parallel processing with configurable worker threads
- **Comprehensive Data Gathering**:
  - Historical price and liquidity data collection
  - Trading volume data from Uniswap Subgraph API
  - Tick-specific liquidity data for accurate fee estimation
  - Impermanent loss calculation
  - Volatility estimation
  - Active ticks analysis
- **Performance Optimizations**:
  - Block grouping by proximity for efficient querying
  - Automatic batch size adjustment based on historical performance
  - Memory-efficient caching with cleanup mechanisms
  - Graceful degradation for rate-limited environments
- **Data Visualization and Export**:
  - Interactive price and liquidity charts
  - Exportable datasets for further analysis
  - Metadata preservation for reproducibility

### Liquidity Strategy
- **Portfolio Value Tracking**:
  - Initial capital configuration
  - Realistic portfolio value calculation over time
  - Support for USD-based performance tracking
- **Rebalancing Cost Models**:
  - Swap slippage based on position size and liquidity
  - Gas cost modeling
  - DEX trading fees
- **Fee Calculations**:
  - Realistic fee assessments based on position-to-pool ratio
  - Volume-based fee estimation
  - Tick-specific liquidity concentration
- **Advanced Performance Metrics**:
  - Maximum drawdown and recovery analysis
  - Sortino ratio (downside risk adjustment)
  - Calmar ratio (return relative to drawdown)
  - Win/loss metrics (win rate, profit factor)
  - Detailed rebalance event analysis
- **Strategy Comparison**:
  - Side-by-side comparison of multiple strategies
  - Conservative, moderate, and aggressive presets
  - Trigger analysis (time vs. price vs. volatility)
- **Visualization**:
  - Portfolio value over time
  - Maximum drawdown periods
  - Performance attribution (fees, IL, costs)
  - Rebalance events by trigger type

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ilyamk/uniswap-v3-lp-strategy-toolkit.git
cd uniswap-v3-lp-strategy-toolkit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Ethereum RPC provider:
   - Get an API key from [Alchemy](https://www.alchemy.com/)
   - Create a `.env` file with your API key:
   ```
   ALCHEMY_API_KEY=your_api_key_here
   THEGRAPH_API_KEY=your_graph_api_key_here  # Optional, for volume data
   ```

## 🛠️ Usage

### Data Collection

The `pool_data.py` script collects historical data from a Uniswap V3 pool with robust RPC management and optimized data collection.

```bash
python pool_data.py --pool <pool_address> --days <days_to_collect> --ticks --volume --export-ticks
```

#### Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--pool` | Uniswap V3 pool address | wstETH/ETH 0.01% pool |
| `--days` | Number of days to collect data for | 365 |
| `--step` | Block step for data collection | 1000 |
| `--output` | Base name for output files | uniswap_v3_data |
| `--ticks` | Include active ticks data (slower) | False |
| `--no-ticks` | Explicitly disable active ticks collection | False |
| `--backtest` | Run basic strategy backtesting | False |
| `--volume` | Fetch trading volume data from Uniswap Subgraph | False |
| `--export-ticks` | Export tick data in a format usable by strategy script | False |
| `--adaptive` | Enable adaptive sampling based on volatility | True |
| `--no-adaptive` | Disable adaptive sampling | False |
| `--parallel` | Enable parallel processing, if you have high RPC rate limit | True |
| `--no-parallel` | Disable parallel processing, if you have low RPC rate limit of free plan | False |
| `--workers` | Number of worker threads for parallel processing | 4 |
| `--batch-size` | Batch size for parallel processing | 100 |
| `--tick-range` | Number of ticks to check on each side of current tick | 10 |
| `--multicall-batch-size` | Batch size for multicall operations | 10 |

#### Examples:

**Basic Data Collection:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 90 --output wsteth_eth_data
```
This collects 90 days of basic price and liquidity data for the wstETH/ETH pool without tick-specific data.

**Comprehensive Data Collection:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 90 --ticks --volume --export-ticks --output wsteth_eth_data
```
This collects 90 days of data including tick-specific liquidity and trading volume, and exports the tick data for use in strategy backtesting.

**High-Performance Collection:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 30 --parallel --workers 8 --batch-size 200 --output high_perf_data
```
This uses parallel processing with 8 worker threads and larger batch sizes to speed up data collection for 30 days of history.

**Detailed Tick Analysis:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 14 --ticks --tick-range 30 --output detailed_tick_data
```
This collects 14 days of data with an expanded tick range (30 ticks on each side of the current price) for more detailed liquidity analysis.

**Low-Resource Collection:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 180 --no-adaptive --no-parallel --step 2000 --output low_resource_data
```
This collects 180 days of data with larger block steps and disabled adaptive sampling and parallel processing, suitable for systems with limited resources.

**Collection with Immediate Backtesting:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 60 --ticks --volume --backtest --output backtest_ready_data
```
This collects 60 days of data and immediately runs a basic strategy backtest on the collected data.

**RPC-Friendly Collection:**
```bash
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 30 --multicall-batch-size 5 --batch-size 50 --output rpc_friendly_data
```
This uses smaller batch sizes for multicall operations and parallel processing to reduce the load on the RPC provider and avoid rate limiting.

### Strategy Backtesting

The `liquidity_strategy.py` script implements and backtests liquidity provision strategies with comprehensive performance metrics.

```bash
python liquidity_strategy.py --data <pool_data_file> --strategy <strategy_type> --initial-capital <starting_usd> --daily-volume <avg_volume>
```

#### Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Pool data CSV file | uniswap_v3_data_wstETH_WETH_0_pool_data.csv |
| `--il` | Impermanent loss data CSV file | uniswap_v3_data_wstETH_WETH_0.01pct_impermanent_loss.csv |
| `--tick-range` | Tick range for liquidity provision | 10 |
| `--interval` | Rebalance interval: 'hourly', 'daily', 'weekly', or seconds | daily |
| `--price-threshold` | Price change threshold for rebalancing in percent | 0.5 |
| `--volatility-threshold` | Volatility threshold for rebalancing | 1.0 |
| `--output` | Base name for output files | strategy_results |
| `--strategy` | Strategy type: 'time' or 'adaptive' | adaptive |
| `--initial-capital` | Initial capital in USD | 100,000 |
| `--daily-volume` | Average daily trading volume in USD | None |
| `--tick-data` | Path to tick data JSON file | None |
| `--gas-price` | Gas price in Gwei for transaction cost calculation | 30.0 |
| `--compare-strategies` | Compare multiple strategy presets | False |

#### Examples:

Time-based strategy with daily rebalancing and initial capital:
```bash
python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --strategy time --interval daily --tick-range 15 --initial-capital 100000 --output time_daily_results
```

Adaptive strategy with realistic fees and trading costs:
```bash
python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --strategy adaptive --interval daily --price-threshold 0.1 --initial-capital 100000 --daily-volume 5000000 --tick-data wsteth_eth_data_tick_data.json --output adaptive_results
```

Compare multiple strategy presets:
```bash
python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --initial-capital 100000 --compare-strategies --output strategy_comparison
```

## 📊 Strategy Types

### Time-Based Strategy
Rebalances positions at fixed time intervals (hourly, daily, weekly, or custom). This strategy is simple but effective in many market conditions.

### Adaptive Strategy
Combines multiple rebalancing triggers:
1. **Time-based**: Rebalance after a fixed time interval
2. **Price-based**: Rebalance when price moves beyond a threshold
3. **Volatility-based**: Rebalance during high volatility periods

Both strategies dynamically adjust the position range based on recent price volatility, which helps optimize capital efficiency.

## 📉 Performance Metrics

The toolkit calculates a comprehensive set of performance metrics:

### Basic Metrics
- **Total Return**: Overall percentage return of the strategy
- **Annualized Return**: Return projected to an annual basis
- **Fees Collected**: Total fees earned from providing liquidity
- **Impermanent Loss**: Loss incurred due to price divergence
- **Net Return**: Fees minus IL and transaction costs

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return using standard deviation
- **Sortino Ratio**: Risk-adjusted return penalizing only downside volatility
- **Calmar Ratio**: Return relative to maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value
- **Drawdown Duration**: Length of maximum drawdown period
- **Recovery Time**: Time taken to recover from maximum drawdown

### Statistical Metrics
- **Win Rate**: Percentage of periods with positive returns
- **Profit Factor**: Ratio of gross profits to gross losses
- **Win/Loss Ratio**: Average win amount divided by average loss amount
- **Average Position Duration**: Average time positions are held

These metrics allow for comprehensive strategy evaluation and comparison.

## 📁 Output Files

### Pool Data Collector
- `<output>_pool_data.csv`: Historical pool data (price, liquidity, volume)
- `<output>_impermanent_loss.csv`: Calculated impermanent loss data
- `<output>_metadata.json`: Pool metadata including token details and collection parameters
- `<output>_plot.png`: Price and liquidity visualization
- `<output>_tick_data.json`: Active ticks data formatted for strategy use
- `<output>_ticks_data.csv`: Detailed tick-specific liquidity data (if collected)

### Liquidity Strategy
- `<output>_rebalance_events.csv`: Details of each rebalancing event including triggers and costs
- `<output>_metrics.json`: Comprehensive strategy performance metrics
- `<output>_plot.png`: Enhanced strategy visualization with portfolio value and performance attribution

## 📈 Analyzing Results

### Key Metrics

The enhanced backtesting includes:

1. **Initial & Final Capital**: Starting capital and final portfolio value
2. **Total Return & Annualized Return**: Overall and annualized performance
3. **Components Breakdown**:
   - Fees Collected
   - Impermanent Loss
   - Rebalancing Costs (swap fees, slippage, gas)
4. **Risk Metrics**:
   - Sharpe, Sortino, and Calmar Ratios
   - Maximum Drawdown (percentage and duration)
5. **Win/Loss Metrics**:
   - Win Rate
   - Profit Factor
   - Win/Loss Ratio
6. **Rebalancing Analysis**:
   - Total Rebalances
   - Trigger Distribution (time, price, volatility)
   - Average Time Between Rebalances

### Rebalance Events Analysis

The `<output>_rebalance_events.csv` file contains detailed information:

| Column | Description |
|--------|-------------|
| `timestamp` | Unix timestamp of the rebalance |
| `datetime` | Human-readable date and time |
| `old_range_low` | Lower tick of previous position |
| `old_range_high` | Upper tick of previous position |
| `new_range_low` | Lower tick of new position |
| `new_range_high` | Upper tick of new position |
| `price` | Token price at rebalance time |
| `trigger` | What triggered the rebalance (time, price, volatility) |
| `time_since_last` | Seconds since previous rebalance |
| `price_change_pct` | Percentage price change since last rebalance |
| `volatility` | Estimated daily volatility at rebalance time |
| `rebalance_cost` | Total cost of the rebalance operation |
| `swap_fee` | DEX trading fee for the rebalance |
| `slippage` | Slippage cost during rebalance |
| `gas_cost` | Gas cost in USD for the transaction |
| `portfolio_value` | Portfolio value after rebalance |

### Enhanced Visualization

The `<output>_plot.png` file provides a comprehensive visual representation:

1. **Price Panel**: Price chart with rebalance events marked by trigger type
2. **Liquidity Panel**: Pool liquidity over time
3. **Portfolio Value Panel**: Portfolio value evolution with drawdown periods
4. **Performance Attribution Panel**: Cumulative returns, fees, IL, and costs

## 🌟 Examples

### Example 1: Complete Data Collection Workflow

```bash
# Collect comprehensive data including volume and tick data
python pool_data.py --pool 0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa --days 30 --ticks --volume --export-ticks --output wsteth_eth_data
```

This command collects 30 days of historical data, trading volume, and tick-specific liquidity.

### Example 2: Realistic Strategy Backtesting

```bash
# Run adaptive strategy with realistic costs
python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --il wsteth_eth_data_impermanent_loss.csv --tick-data wsteth_eth_data_tick_data.json --strategy adaptive --initial-capital 10000 --daily-volume 5000000 --gas-price 25 --output realistic_strategy
```

This performs backtesting with realistic modeling of fees, slippage, and gas costs.

### Example 3: Strategy Comparison

```bash
# Compare multiple strategy presets
python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --initial-capital 100000 --compare-strategies --output strategy_comparison
```

This runs backtests on conservative, moderate, and aggressive strategy presets and provides a comparison table of results.

### Example 4: Optimizing Parameters

```bash
# Optimize tick range for specific market conditions
for range in 5 10 15 20 30; do
  python liquidity_strategy.py --data wsteth_eth_data_0.01_pool_data.csv --strategy adaptive --tick-range $range --initial-capital 100000 --output range_${range}
done
```

Run multiple backtests with different parameters to find optimal settings.

## 📝 Notes for future code improvements

- Optimize the RPC Alchemy calls for data collection or use the OKU API instead.
- Optimize the RPC rate limits on paid and public plans.
- Add a correct fee tier for the analysed pool in strategy simulation [now it's hardcoded as 0.001]
- Fix the fee calculation in the strategy simulation based on the correct [now it's only count the fees based on the collected tick data, that is not correct]
- Add more strategies presets and more dynamic metrics to backtest
- Implement AI Agent to automatically choose the best strategy and parameters based on the market conditions
- Improve multicall batching with more intelligent batch size adjustment
- Add support for more Uniswap V3 pools and other DEXes
- Implement position range optimization based on historical liquidity distribution
- Create a web interface for easier strategy visualization and comparison

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request to this [repository](https://github.com/ilyamk/uniswap-v3-lp-strategy-toolkit).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📚 Additional Resources

- [Uniswap V3 Whitepaper](https://uniswap.org/whitepaper-v3.pdf)
- [Uniswap V3 Documentation](https://docs.uniswap.org/concepts/overview)
- [Risks and Returns of Uniswap V3 Liquidity Providers](https://liobaheimba.ch/assets/pdf/Papers/Risks_and_Returns_of_Uniswap_V3_Liquidity_Providers.pdf)