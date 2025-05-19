# Generative AI API Cost Calculator (Streamlit Prototype)

**PROJECT SPECIFICATION**

## Introduction

This document outlines the specifications for a prototype Generative AI API Cost Calculator, to be developed using Python and the Streamlit framework. The calculator will allow users to input various parameters related to their organization's potential usage of Generative AI APIs and visualize the resulting cost estimates. This tool is designed to facilitate informed decision-making regarding the adoption of Generative AI technologies by providing a transparent and customizable cost model.

## Goals

*   Enable easy cost comparison between per-seat licensing models and direct API usage.
*   Provide a flexible and customizable environment to explore different usage scenarios.
*   Support multiple Generative AI models and their respective pricing structures.
*   Visualize cost projections clearly and intuitively to aid decision-making.

## Target Audience

*   IT Budget Planners
*   Project Managers
*   Software Engineers
*   Decision-Makers Evaluating Generative AI Adoption

## Scope

The calculator will focus on estimating the cost of Generative AI API usage based on token consumption. It will include the following features:

*   User Archetype Definition: Allow users to define different user archetypes within an organization (e.g., "Explorers," "Creators," "Builders") and specify their average daily/monthly API usage.
*   Model Selection: Support a selection of popular Generative AI models (e.g., OpenAI GPT-4o, GPT-3.5 Turbo, Anthropic Claude 3 Haiku, Google Gemini 1.5 Flash) with their corresponding input/output token pricing.
*   Customizable Parameters: Allow users to adjust parameters such as the average number of prompts per day, average input/output token lengths, and the distribution of model usage across different archetypes.
*   Cost Visualization: Display the estimated monthly cost per user, total monthly cost for the organization, and cost savings compared to a per-seat licensing model.
*   Scenario Planning: Enable users to create and compare different usage scenarios by adjusting the input parameters.
*   Data Export: Allow users to export the calculated cost estimates as a CSV file or similar format.

## Functional Requirements

### Input Parameters

The calculator should accept the following input parameters:

*   Organization Size: Total number of employees.
*   User Archetypes:
    *   Name (e.g., Explorer, Creator, Builder)
    *   Percentage of employees belonging to this archetype
    *   Average prompts per day
    *   Average input tokens per prompt
    *   Average output tokens per prompt
*   Generative AI Models:
    *   Selectable from a list (e.g., GPT-4o, GPT-3.5 Turbo, Claude 3 Haiku, Gemini 1.5 Flash)
    *   Input token price (USD per 1k or 1M tokens)
    *   Output token price (USD per 1k or 1M tokens)
    *   Option to add custom models with user-defined pricing
*   Model Usage Distribution: Percentage of prompts directed to each model (allowing for a mixed-model approach).
*   Per-Seat License Cost: Monthly cost per user for a per-seat licensing model.
*   Working Days per Month: Number of working days in a month.

### Calculations

The calculator should perform the following calculations:

*   Daily Tokens per User Archetype: (Prompts per day) * (Input tokens per prompt + Output tokens per prompt)
*   Monthly Tokens per User Archetype: (Daily Tokens per User Archetype) * (Working Days per Month)
*   Monthly Cost per User Archetype (per model): (Monthly Tokens per User Archetype) * (Model Usage Percentage) * (Token Price per 1k/1M Tokens)
*   Total Monthly Cost per User Archetype: Sum of Monthly Cost per User Archetype across all models.
*   Weighted Average Monthly Cost per User: Sum of (Total Monthly Cost per User Archetype) * (Percentage of Employees in Archetype)
*   Total Monthly API Cost: (Weighted Average Monthly Cost per User) * (Organization Size)
*   Total Monthly Per-Seat License Cost: (Per-Seat License Cost) * (Organization Size)
*   Cost Savings (API vs. Per-Seat): (Total Monthly Per-Seat License Cost) - (Total Monthly API Cost)
*   Cost Per Token (for debugging purposes)

### Output Visualization

The calculator should display the following outputs:

*   Table summarizing input parameters for each user archetype and AI model.
*   Table showing calculated monthly token usage and cost per user archetype.
*   Total monthly API cost for the organization.
*   Total monthly per-seat license cost for the organization.
*   Estimated monthly cost savings by using the API approach.
*   Clear graphical representation (e.g., bar chart, pie chart) comparing the API cost to the per-seat licensing cost.
*   A summary panel showcasing the most important metrics: total API cost, total per-seat cost, and the difference between the two.

## Non-Functional Requirements

*   Usability: The calculator should be easy to use and intuitive, with clear instructions and helpful tooltips.
*   Performance: Calculations should be performed quickly and efficiently, with minimal latency.
*   Scalability: The calculator should be able to handle a large number of users and complex usage scenarios.
*   Maintainability: The code should be well-structured and documented to facilitate future maintenance and enhancements.
*   Responsiveness: The application should be responsive and display correctly on various screen sizes and devices.

## Technology Stack

*   Programming Language: Python
*   Web Framework: Streamlit
*   Data Visualization: Streamlit's built-in charting capabilities or libraries like Matplotlib, Plotly, or Altair (as needed for advanced visualizations)
*   Data Export: Pandas (for creating CSV files)

## User Interface (UI) Design

The UI should be organized and intuitive, with clear sections for input parameters, calculations, and output visualization. Consider the following layout:

*   Sidebar: Navigation and global settings (e.g., organization size, per-seat license cost).
*   Main Panel:
    *   Tabbed interface to separate User Archetype definition, Model selection, and Results.
    *   Clear labeling and tooltips for all input fields.
    *   Dynamic updates to calculations as input parameters are changed.
    *   Visualizations that effectively communicate the cost comparison.

## Prototype Development Process

*   Phase 1: Core Functionality
    *   Implement basic input parameters and calculations.
    *   Display results in a tabular format.
*   Phase 2: UI Enhancement
    *   Improve the UI with a Streamlit layout, clear labels, and tooltips.
    *   Add data visualization using Streamlit charts.
*   Phase 3: Advanced Features
    *   Implement scenario planning and data export functionality.
    *   Add support for custom Generative AI models.

## Testing

*   Unit tests to verify the correctness of the calculations.
*   User acceptance testing (UAT) to ensure the calculator meets the needs of the target audience.

## Future Enhancements (Out of Scope for Prototype)

*   Integration with actual API usage data for more accurate cost estimates.
*   Advanced scenario modeling, including the impact of fine-tuning and RAG.
*   Support for different pricing models (e.g., pay-per-request, reserved capacity).
*   Integration with corporate budgeting tools.

## Conclusion

This prototype project specification provides a detailed roadmap for developing a Generative AI API Cost Calculator using Python and Streamlit. By following these specifications, the development team can create a valuable tool for organizations to assess the cost-effectiveness of adopting Generative AI technologies.