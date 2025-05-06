# Planning and Strategy AGI

A sophisticated multi-agent system for financial market analysis and trading strategy development. This project implements an AGI (Artificial General Intelligence) approach to market analysis, combining multiple specialized agents to propose, validate, and execute trading strategies.

## Overview

This system employs a multi-agent architecture where different agents specialize in various aspects of market analysis and trading strategy development. The agents work together through a Socratic dialogue process to critique and refine trading strategies before execution. The system uses GPT-4 for intelligent decision-making and strategy generation.

## Key Components

### Strategy Agents
- **MomentumAgent**: 
  - Analyzes price acceleration and volume trends
  - Uses GPT-4 for intelligent portfolio allocation
  - Implements momentum-based trading strategies
  - Features fallback mechanisms and plan validation

- **MeanReversionAgent**: 
  - Focuses on mean reversion trading strategies
  - Identifies overbought/oversold conditions
  - Generates contrarian trading signals

- **EventDrivenAgent**: 
  - Analyzes market events and news impact
  - Processes real-time market data
  - Generates event-based trading signals

### Support Agents
- **ValidatorAgent**: 
  - Ensures proposed strategies meet predefined constraints
  - Validates portfolio allocations
  - Checks for compliance with trading rules

- **MetaPlannerAgent**: 
  - Coordinates multiple strategy proposals
  - Selects optimal strategy based on market conditions
  - Implements strategy fusion and optimization

- **ExecutorAgent**: 
  - Handles trade execution
  - Manages order routing
  - Implements execution algorithms

- **MemoryAgent**: 
  - Maintains historical performance data
  - Tracks strategy effectiveness
  - Implements learning from past trades

- **PostTradeAnalyzerAgent**: 
  - Evaluates strategy performance
  - Calculates key performance metrics
  - Generates performance reports

### Core Modules
- `agents/`: 
  - Base agent implementations
  - Strategy-specific agents
  - Meta-planning and coordination

- `data/`: 
  - Market data loading and processing
  - Real-time data feeds
  - Historical data management

- `execution/`: 
  - Trade execution logic
  - Order management
  - Position tracking

- `evaluation/`: 
  - Performance metrics calculation
  - Strategy analysis tools
  - Risk assessment

- `memory/`: 
  - Historical data storage
  - Performance tracking
  - Learning mechanisms

- `simulation/`: 
  - Backtesting framework
  - Strategy simulation
  - Performance testing

- `utils/`: 
  - Helper functions
  - Common utilities
  - Data processing tools

- `visualize/`: 
  - Performance charts
  - Strategy visualization
  - Market data plots

## Features

### Strategy Development
- Multi-agent strategy generation
- GPT-4 powered decision making
- Socratic dialogue for strategy refinement
- Constraint validation and compliance checking
- Portfolio optimization

### Analysis and Learning
- Real-time market analysis
- Performance tracking and evaluation
- Historical data analysis
- Strategy effectiveness metrics
- Continuous learning from past performance

### Risk Management
- Portfolio risk assessment
- Position sizing optimization
- Risk-adjusted return calculation
- Drawdown analysis
- Volatility management

### Execution
- Smart order routing
- Execution algorithm implementation
- Position management
- Real-time trade monitoring
- Performance tracking

## Simulation Workflow & Features

- **Modern Web UI (Streamlit):**
  - Configure simulation parameters (strategy, dates, capital, tickers) interactively
  - Live progress bar and real-time portfolio value updates
  - Visual breakdown of current investments and allocations at each step
  - Final results and metrics displayed at the top after simulation
  - Agent comparison table for all windows, with clear meta-planner selection
  - Robust error handling and Arrow compatibility for smooth UI experience

- **Ensemble Mode with Meta-Planner:**
  - All agents (Momentum, Mean Reversion, Event Driven) run in parallel for an initial evaluation period (default: 2 windows)
  - Meta-planner evaluates agent performance (cumulative return) during the evaluation period
  - After evaluation, the best-performing agent is selected and used for the remainder of the simulation
  - Full window-by-window agent comparison and final allocation charts

- **Data Pipeline:**
  - Uses Polygon.io for historical market data (OHLCV, VWAP, dividends, splits)
  - Efficient caching and lookback support

- **Extensible & Modular:**
  - Easily add new agent strategies, meta-planner logic, or evaluation metrics
  - Modular design for rapid experimentation

## How to Use the Web UI

1. Run the Streamlit app:
```bash
streamlit run app.py
```
2. Open the provided local URL in your browser
3. Configure your simulation parameters in the sidebar
4. Click "Run Simulation" to start
5. View live progress, agent performance, and final results in the main panel

## New Features (2024)
- Ensemble mode with meta-planner evaluation period
- Modern Streamlit UI with live progress, agent comparison, and final results at the top
- Robust error handling and Arrow compatibility
- Polygon.io data integration (replaces yfinance)
- Evaluation period logic for agent selection
- Improved portfolio value and allocation breakdowns
- Full window-by-window agent comparison table

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Planning_and_Strategy_AGI.git
cd Planning_and_Strategy_AGI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file with your OpenAI API key
   - Add any other required API keys or configurations

## Usage

1. Run the main simulation:
```bash
python main.py
```

2. For specific demonstrations:
```bash
python demo_run.py
```

3. Run custom simulations:
```bash
python run_simulation.py
```

## Dependencies

- openai: GPT-4 integration
- yfinance: Market data access
- python-dotenv: Environment management
- pandas: Data manipulation
- tqdm: Progress tracking
- chromadb: Vector storage
- matplotlib: Data visualization
- seaborn: Statistical visualization

## Project Structure

```
Planning_and_Strategy_AGI/
├── agents/           # Agent implementations
│   ├── strategy/    # Trading strategy agents
│   └── base.py      # Base agent class
├── data/            # Market data handling
├── execution/       # Trade execution
├── evaluation/      # Performance analysis
├── memory/         # Historical data tracking
├── simulation/     # Backtesting
├── utils/          # Utility functions
├── visualize/      # Visualization tools
├── main.py         # Main entry point
├── demo_run.py     # Demonstration script
├── run_simulation.py # Simulation runner
└── requirements.txt # Project dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. When contributing, please ensure:

1. Code follows the existing style and structure
2. New features include appropriate tests
3. Documentation is updated
4. Changes are properly documented in commit messages

## License

[Add your license information here]

## Contact

[Add your contact information here] 