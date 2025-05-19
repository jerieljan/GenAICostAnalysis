# AI API vs AI Team Plans Calculator

A Streamlit-based tool for calculating and comparing costs between per-seat licensing models and direct API usage for
Generative AI services.

## Overview

This calculator helps organizations make informed decisions about their Generative AI adoption strategy by providing
detailed cost analysis and comparisons. It supports multiple user archetypes, various AI models, and flexible usage
scenarios.

## Features

- **User Archetype Analysis**: Define and analyze different user types (e.g., Explorers, Creators, Builders)
- **Multiple Model Support**: Compare costs across various AI models (GPT-4, GPT-3.5 Turbo, Claude 3 Haiku, Gemini 1.5
  Flash)
- **Customizable Parameters**: Adjust usage patterns, token counts, and organizational parameters
- **Cost Visualization**: Interactive charts and tables for cost analysis
- **Scenario Planning**: Create and compare different usage scenarios
- **Data Export**: Export calculations to CSV format

## Installation

1. Clone the repository:

```
bash
git clone <repository-url>
```

2. Install dependencies:

```
bash
pip install -r requirements.txt
```

3. Run the application:

```
bash
streamlit run app.py
```

## Usage

1. Set your organization size and global parameters in the sidebar
2. Define user archetypes and their usage patterns
3. Select and configure AI models
4. View calculated costs and comparisons in the results tab
5. Export results as needed

## Requirements

- Python 3.x
- Key dependencies:
    - streamlit>=1.45.1
    - pandas>=2.2.3
    - plotly>=6.1.0
    - numpy>=2.2.6

## Target Audience

- IT Budget Planners
- Project Managers
- Software Engineers
- Decision-Makers evaluating Generative AI adoption

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Enhancements

- Integration with actual API usage data
- Advanced scenario modeling with fine-tuning and RAG support
- Support for additional pricing models
- Integration with corporate budgeting tools