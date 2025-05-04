# Planning and Strategy AGI

A sophisticated multi-agent system for financial market analysis and trading strategy development. This project implements an AGI (Artificial General Intelligence) approach to market analysis, combining multiple specialized agents to propose, validate, and execute trading strategies.

## Overview

This system employs a multi-agent architecture where different agents specialize in various aspects of market analysis and trading strategy development. The agents work together through a Socratic dialogue process to critique and refine trading strategies before execution.

## Key Components

### Agents
- **MomentumAgent**: Identifies and capitalizes on market momentum trends
- **MeanReversionAgent**: Focuses on mean reversion trading strategies
- **EventDrivenAgent**: Analyzes and responds to market events and news
- **ValidatorAgent**: Ensures proposed strategies meet predefined constraints
- **MetaPlannerAgent**: Coordinates and selects the best strategy from multiple proposals
- **ExecutorAgent**: Handles the execution of selected trading strategies
- **MemoryAgent**: Maintains historical data and learning from past performance
- **PostTradeAnalyzerAgent**: Evaluates strategy performance post-execution

### Core Modules
- `agents/`: Contains all agent implementations
- `data/`: Market data loading and processing
- `execution/`: Trade execution logic
- `evaluation/`: Performance analysis tools
- `memory/`: Historical data and learning mechanisms
- `simulation/`: Backtesting and simulation capabilities
- `utils/`: Utility functions and helper modules
- `visualize/`: Data visualization tools

## Features

- Multi-agent strategy development
- Socratic dialogue for strategy refinement
- Constraint validation and compliance checking
- Performance analysis and learning
- Historical data tracking and analysis
- Market data visualization

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

## Usage

1. Set up your environment variables:
   - Create a `.env` file with necessary API keys and configurations

2. Run the main simulation:
```bash
python main.py
```

3. For specific demonstrations:
```bash
python demo_run.py
```

## Dependencies

- openai
- yfinance
- python-dotenv
- pandas
- tqdm
- chromadb
- matplotlib
- seaborn

## Project Structure

```
Planning_and_Strategy_AGI/
├── agents/           # Agent implementations
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

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here] 