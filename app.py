import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="AI API vs AI Team Plans Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    .positive {
        color: #4CAF50;
    }
    .negative {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="main-header">AI API vs AI Team Plans Calculator</div>', unsafe_allow_html=True)
st.markdown("""
This calculator helps you estimate and compare the costs of using Generative AI APIs versus per-seat licensing models.
Input your organization's parameters and see detailed cost projections to make informed decisions.
""")

# Initialize session state for storing user archetypes and models
if 'user_archetypes' not in st.session_state:
    st.session_state.user_archetypes = [
        {
            'name': 'Explorer',
            'percentage': 50,
            'prompts_per_day': 5,
            'input_tokens_per_prompt': 500,
            'output_tokens_per_prompt': 1000
        },
        {
            'name': 'Developer',
            'percentage': 20,
            'prompts_per_day': 15,
            'input_tokens_per_prompt': 1000,
            'output_tokens_per_prompt': 3000
        },
        {
            'name': 'Builder A',
            'percentage': 5,
            'prompts_per_day': 20,
            'input_tokens_per_prompt': 5000,
            'output_tokens_per_prompt': 10000
        },
        {
            'name': 'Builder B',
            'percentage': 5,
            'prompts_per_day': 50,
            'input_tokens_per_prompt': 2000,
            'output_tokens_per_prompt': 5000
        },
        {
            'name': 'Agent',
            'percentage': 3,
            'prompts_per_day': 5,
            'input_tokens_per_prompt': 50000,
            'output_tokens_per_prompt': 100000
        },

    ]

if 'ai_models' not in st.session_state:
    st.session_state.ai_models = [
        {
            'name': 'Gemini 2.0 Flash',
            'input_price_per_1M': 0.15,
            'output_price_per_1M': 0.60,
            'usage_percentage': 30
        },
        {
            'name': 'Gemini 2.5 Pro Preview',
            'input_price_per_1M': 1.25,
            'output_price_per_1M': 10,
            'usage_percentage': 25
        },
        {
            'name': 'Claude 3.5 Sonnet',
            'input_price_per_1M': 3,
            'output_price_per_1M': 15,
            'usage_percentage': 20
        },
        {
            'name': 'GPT-4.1',
            'input_price_per_1M': 2,
            'output_price_per_1M': 8,
            'usage_percentage': 15
        },
        {
            'name': 'o4-mini',
            'input_price_per_1M': 1.10,
            'output_price_per_1M': 4.40,
            'usage_percentage': 10
        }
    ]

# Function to download data as CSV
def get_csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Sidebar - Global Settings
st.sidebar.markdown('### Global Settings')

# Organization size
organization_size = st.sidebar.number_input(
    "Organization Size (Number of Employees)",
    min_value=1,
    max_value=1000000,
    value=1000,
    help="Total number of employees in your organization"
)

# Per-seat license cost
per_seat_license_cost = st.sidebar.number_input(
    "Per-Seat License Cost (USD/month)",
    min_value=0.0,
    max_value=10000.0,
    value=25.0,
    step=1.0,
    help="Monthly cost per user for a per-seat licensing model"
)

# Working days per month
working_days = st.sidebar.number_input(
    "Working Days per Month",
    min_value=1,
    max_value=31,
    value=22,
    help="Number of working days in a month"
)

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["User Archetypes", "AI Models", "Results", "Export"])

# Tab 1: User Archetypes
with tab1:
    st.markdown('<div class="sub-header">User Archetypes</div>', unsafe_allow_html=True)
    st.markdown("Define different user archetypes within your organization and their usage patterns.")

    # Add new archetype button
    if st.button("Add New User Archetype"):
        st.session_state.user_archetypes.append({
            'name': f'Archetype {len(st.session_state.user_archetypes) + 1}',
            'percentage': 0,
            'prompts_per_day': 10,
            'input_tokens_per_prompt': 300,
            'output_tokens_per_prompt': 1200
        })

    # Display and edit user archetypes
    archetypes_to_remove = []

    for i, archetype in enumerate(st.session_state.user_archetypes):
        col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 0.5], vertical_alignment="bottom")

        with col1:
            st.session_state.user_archetypes[i]['name'] = st.text_input(
                f"Name #{i+1}",
                value=archetype['name'],
                key=f"name_{i}"
            )

        with col2:
            st.session_state.user_archetypes[i]['percentage'] = st.number_input(
                f"% of Employees #{i+1}",
                min_value=0,
                max_value=100,
                value=archetype['percentage'],
                key=f"percentage_{i}"
            )

        with col3:
            st.session_state.user_archetypes[i]['prompts_per_day'] = st.number_input(
                f"Prompts/Day #{i+1}",
                min_value=0,
                max_value=1000,
                value=archetype['prompts_per_day'],
                key=f"prompts_{i}"
            )

        with col4:
            st.session_state.user_archetypes[i]['input_tokens_per_prompt'] = st.number_input(
                f"Input Tokens/Prompt #{i+1}",
                min_value=0,
                max_value=100000,
                value=archetype['input_tokens_per_prompt'],
                key=f"input_tokens_{i}"
            )

        with col5:
            st.session_state.user_archetypes[i]['output_tokens_per_prompt'] = st.number_input(
                f"Output Tokens/Prompt #{i+1}",
                min_value=0,
                max_value=100000,
                value=archetype['output_tokens_per_prompt'],
                key=f"output_tokens_{i}"
            )

        with col6:
            if st.button("🗑️", key=f"remove_archetype_{i}"):
                archetypes_to_remove.append(i)

    # Remove archetypes marked for deletion
    for index in sorted(archetypes_to_remove, reverse=True):
        st.session_state.user_archetypes.pop(index)

    # Validate total percentage
    total_percentage = sum(archetype['percentage'] for archetype in st.session_state.user_archetypes)
    if total_percentage > 100:
        st.warning(f"Total percentage of employees across all archetypes cannot exceed 100%. Current total: {total_percentage}%")
    elif total_percentage < 100:
        non_user_percentage = 100 - total_percentage
        st.info(f"Total percentage of employees across all archetypes is {total_percentage}%. The remaining {non_user_percentage}% represents non-users who will be counted for license costs but not for API costs.")

# Tab 2: AI Models
with tab2:
    st.markdown('<div class="sub-header">AI Models</div>', unsafe_allow_html=True)
    st.markdown("Configure the AI models your organization uses and their pricing.")

    # Add new model button
    if st.button("Add Custom AI Model"):
        st.session_state.ai_models.append({
            'name': f'Custom Model {len(st.session_state.ai_models) - 3}',
            'input_price_per_1M': 1.0,
            'output_price_per_1M': 3.0,
            'usage_percentage': 0
        })

    # Display and edit AI models
    models_to_remove = []

    for i, model in enumerate(st.session_state.ai_models):
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1, 0.5], vertical_alignment="bottom")

        with col1:
            st.session_state.ai_models[i]['name'] = st.text_input(
                f"Model Name #{i+1}",
                value=model['name'],
                key=f"model_name_{i}"
            )

        with col2:
            st.session_state.ai_models[i]['input_price_per_1M'] = st.number_input(
                f"Input Price (USD/1M tokens) #{i+1}",
                min_value=0.0,
                max_value=1000.0,
                value=float(model['input_price_per_1M']),
                format="%.2f",
                key=f"input_price_{i}"
            )

        with col3:
            st.session_state.ai_models[i]['output_price_per_1M'] = st.number_input(
                f"Output Price (USD/1M tokens) #{i+1}",
                min_value=0.0,
                max_value=1000.0,
                value=float(model['output_price_per_1M']),
                format="%.2f",
                key=f"output_price_{i}"
            )

        with col4:
            st.session_state.ai_models[i]['usage_percentage'] = st.number_input(
                f"Usage % #{i+1}",
                min_value=0,
                max_value=100,
                value=model['usage_percentage'],
                key=f"usage_percentage_{i}"
            )

        with col5:
            if i > 0:  # Allow removing all models except the first one
                if st.button("🗑️", key=f"remove_model_{i}"):
                    models_to_remove.append(i)
            else:
                st.write("")  # Empty space for alignment

    # Remove models marked for deletion
    for index in sorted(models_to_remove, reverse=True):
        st.session_state.ai_models.pop(index)

    # Validate total percentage
    total_model_percentage = sum(model['usage_percentage'] for model in st.session_state.ai_models)
    if total_model_percentage != 100:
        st.warning(f"Total usage percentage across all models should be 100%. Current total: {total_model_percentage}%")

# Tab 3: Results
with tab3:
    st.markdown('### Cost Analysis Results')

    # Perform calculations
    # 1. Calculate daily and monthly tokens per user archetype
    results = []

    for archetype in st.session_state.user_archetypes:
        daily_tokens = archetype['prompts_per_day'] * (archetype['input_tokens_per_prompt'] + archetype['output_tokens_per_prompt'])
        monthly_tokens = daily_tokens * working_days

        # Calculate cost per model for this archetype
        archetype_model_costs = []
        for model in st.session_state.ai_models:
            # Calculate the percentage of this archetype's tokens that go to this model
            model_usage_fraction = model['usage_percentage'] / 100

            # Calculate input and output tokens for this model
            model_monthly_input_tokens = monthly_tokens * (archetype['input_tokens_per_prompt'] / (archetype['input_tokens_per_prompt'] + archetype['output_tokens_per_prompt'])) * model_usage_fraction
            model_monthly_output_tokens = monthly_tokens * (archetype['output_tokens_per_prompt'] / (archetype['input_tokens_per_prompt'] + archetype['output_tokens_per_prompt'])) * model_usage_fraction

            # Calculate cost for this model
            model_input_cost = model_monthly_input_tokens * (model['input_price_per_1M'] / 1000000)
            model_output_cost = model_monthly_output_tokens * (model['output_price_per_1M'] / 1000000)
            model_total_cost = model_input_cost + model_output_cost

            archetype_model_costs.append({
                'model_name': model['name'],
                'input_tokens': model_monthly_input_tokens,
                'output_tokens': model_monthly_output_tokens,
                'input_cost': model_input_cost,
                'output_cost': model_output_cost,
                'total_cost': model_total_cost
            })

        # Calculate total cost for this archetype across all models
        total_archetype_cost = sum(model_cost['total_cost'] for model_cost in archetype_model_costs)

        results.append({
            'archetype_name': archetype['name'],
            'percentage': archetype['percentage'],
            'daily_tokens': daily_tokens,
            'monthly_tokens': monthly_tokens,
            'model_costs': archetype_model_costs,
            'total_cost': total_archetype_cost
        })

    # 2. Calculate weighted average monthly cost per user
    weighted_avg_cost_per_user = sum(result['total_cost'] * (result['percentage'] / 100) for result in results)

    # 3. Calculate total monthly costs
    # Calculate the percentage of employees who are actually users
    total_user_percentage = sum(archetype['percentage'] for archetype in st.session_state.user_archetypes)
    # Only count actual users for API costs
    total_monthly_api_cost = weighted_avg_cost_per_user * (total_user_percentage / 100 * organization_size)
    # Count all employees for license costs
    total_monthly_per_seat_cost = per_seat_license_cost * organization_size
    cost_savings = total_monthly_per_seat_cost - total_monthly_api_cost

    # Display summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Monthly API Cost", f"${total_monthly_api_cost:,.5f}")

    with col2:
        st.metric("Total Monthly Per-Seat Cost", f"${total_monthly_per_seat_cost:,.2f}")

    with col3:
        savings_color = "positive" if cost_savings > 0 else "negative"
        savings_prefix = "+" if cost_savings > 0 else ""
        st.markdown(f"""
        <h3>Cost Savings</h3>
        <h2 class="{savings_color}">{savings_prefix}${cost_savings:,.2f}</h2>
        <p>({savings_prefix}{(cost_savings/total_monthly_per_seat_cost*100 if total_monthly_per_seat_cost > 0 else 0):,.1f}%)</p>
        """, unsafe_allow_html=True)


    # Display cost comparison chart
    st.markdown('<div class="sub-header">Cost Comparison</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["API-based Cost", "Per-Seat License Cost"],
        y=[total_monthly_api_cost, total_monthly_per_seat_cost],
        marker_color=['#1E88E5', '#FFC107']
    ))
    fig.update_layout(
        title="Monthly Cost Comparison",
        xaxis_title="Cost Model",
        yaxis_title="Monthly Cost (USD)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display detailed results tables
    st.markdown('<div class="sub-header">User Archetype Cost Breakdown</div>', unsafe_allow_html=True)

    # Create a DataFrame for archetype summary
    archetype_summary = pd.DataFrame([
        {
            'Archetype': result['archetype_name'],
            'Percentage of Employees': f"{result['percentage']}%",
            'Daily Tokens': f"{result['daily_tokens']:,}",
            'Monthly Tokens': f"{result['monthly_tokens']:,}",
            'Monthly Cost per User': f"${result['total_cost']:.5f}"
        }
        for result in results
    ])

    st.dataframe(archetype_summary, use_container_width=True)
    # Create a DataFrame for model usage by archetype
    model_usage_rows = []
    for result in results:
        for model_cost in result['model_costs']:
            model_usage_rows.append({
                'Archetype': result['archetype_name'],
                'Model': model_cost['model_name'],
                'Monthly Input Tokens': f"{model_cost['input_tokens']:,.0f}",
                'Monthly Output Tokens': f"{model_cost['output_tokens']:,.0f}",
                'Input Cost': f"${model_cost['input_cost']:.5f}",
                'Output Cost': f"${model_cost['output_cost']:.5f}",
                'Total Cost': f"${model_cost['total_cost']:.5f}"
            })

    model_usage_df = pd.DataFrame(model_usage_rows)

    st.markdown('<div class="sub-header">Model Usage by Archetype</div>', unsafe_allow_html=True)
    st.dataframe(model_usage_df, use_container_width=True)

    # Display cost breakdown by model
    st.markdown('<div class="sub-header">Cost Breakdown by Model</div>', unsafe_allow_html=True)

    # Aggregate costs by model
    model_costs = {}

    # First pass: calculate the total cost without scaling
    total_unscaled_model_cost = 0
    for result in results:
        for model_cost in result['model_costs']:
            model_name = model_cost['model_name']
            if model_name not in model_costs:
                model_costs[model_name] = {
                    'input_cost': 0,
                    'output_cost': 0,
                    'total_cost': 0,
                    'unscaled_input_cost': 0,
                    'unscaled_output_cost': 0,
                    'unscaled_total_cost': 0
                }

            # Weight by percentage of employees
            weight = result['percentage'] / 100 * organization_size

            # Store unscaled costs
            unscaled_input_cost = model_cost['input_cost'] * weight
            unscaled_output_cost = model_cost['output_cost'] * weight
            unscaled_total_cost = model_cost['total_cost'] * weight

            model_costs[model_name]['unscaled_input_cost'] += unscaled_input_cost
            model_costs[model_name]['unscaled_output_cost'] += unscaled_output_cost
            model_costs[model_name]['unscaled_total_cost'] += unscaled_total_cost

            total_unscaled_model_cost += unscaled_total_cost

    # Calculate scaling factor to ensure total matches total_monthly_api_cost
    model_scaling_factor = 1.0
    if total_unscaled_model_cost > 0:  # Avoid division by zero
        model_scaling_factor = total_monthly_api_cost / total_unscaled_model_cost

    # Apply scaling factor
    for model_name in model_costs:
        model_costs[model_name]['input_cost'] = model_costs[model_name]['unscaled_input_cost'] * model_scaling_factor
        model_costs[model_name]['output_cost'] = model_costs[model_name]['unscaled_output_cost'] * model_scaling_factor
        model_costs[model_name]['total_cost'] = model_costs[model_name]['unscaled_total_cost'] * model_scaling_factor

    # Create pie chart for model cost distribution
    model_names = list(model_costs.keys())
    model_total_costs = [model_costs[name]['total_cost'] for name in model_names]

    fig = px.pie(
        values=model_total_costs,
        names=model_names,
        title="Cost Distribution by Model",
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Create a DataFrame for model cost summary
    model_cost_summary = pd.DataFrame([
        {
            'Model': name,
            'Input Cost': f"${model_costs[name]['input_cost']:,.2f}",
            'Output Cost': f"${model_costs[name]['output_cost']:,.2f}",
            'Total Cost': f"${model_costs[name]['total_cost']:,.2f}",
            'Percentage of Total': f"{(model_costs[name]['total_cost'] / total_monthly_api_cost * 100 if total_monthly_api_cost > 0 else 0):,.1f}%"
        }
        for name in model_names
    ])

    st.dataframe(model_cost_summary, use_container_width=True)

    # Display monthly cost breakdown by archetype and model
    st.markdown('<div class="sub-header">Monthly Cost Breakdown by Archetype and Model</div>', unsafe_allow_html=True)

    # Create a DataFrame for monthly cost breakdown by archetype and model
    monthly_breakdown_rows = []

    # First pass: calculate the total cost without scaling
    total_unscaled_cost = 0
    for result in results:
        archetype_percentage = result['percentage']
        archetype_weight = archetype_percentage / 100 * organization_size

        for model_cost in result['model_costs']:
            monthly_cost = model_cost['total_cost'] * archetype_weight
            total_unscaled_cost += monthly_cost

    # Calculate scaling factor to ensure total matches total_monthly_api_cost
    scaling_factor = 1.0
    if total_unscaled_cost > 0:  # Avoid division by zero
        scaling_factor = total_monthly_api_cost / total_unscaled_cost

    # Second pass: apply the scaling factor
    for result in results:
        archetype_name = result['archetype_name']
        archetype_percentage = result['percentage']
        archetype_weight = archetype_percentage / 100 * organization_size

        for model_cost in result['model_costs']:
            model_name = model_cost['model_name']
            monthly_cost = model_cost['total_cost'] * archetype_weight * scaling_factor

            monthly_breakdown_rows.append({
                'Archetype': archetype_name,
                'Model': model_name,
                'Employees': f"{archetype_weight:.0f}",
                'Monthly Cost': f"${monthly_cost:.2f}"
            })

    monthly_breakdown_df = pd.DataFrame(monthly_breakdown_rows)

    # Add a pivot table view for better visualization
    pivot_df = pd.DataFrame(monthly_breakdown_rows)
    pivot_df['Monthly Cost'] = pivot_df['Monthly Cost'].str.replace('$', '').astype(float)
    pivot_table = pd.pivot_table(
        pivot_df, 
        values='Monthly Cost', 
        index=['Archetype'], 
        columns=['Model'], 
        aggfunc='sum',
        fill_value=0
    )

    # Format the pivot table values as currency
    formatted_pivot = pivot_table.applymap(lambda x: f"${x:.2f}")

    # Add row totals
    row_totals = pivot_table.sum(axis=1)
    formatted_pivot['Total'] = row_totals.apply(lambda x: f"${x:.2f}")

    # Add column totals
    col_totals = pivot_table.sum(axis=0)
    total_row = col_totals.apply(lambda x: f"${x:.2f}")
    total_row['Total'] = f"${col_totals.sum():.2f}"
    formatted_pivot.loc['Total'] = total_row

    st.dataframe(formatted_pivot, use_container_width=True)

    # Verify that the total matches the Total Monthly API Cost
    total_from_pivot = col_totals.sum()
    if abs(total_from_pivot - total_monthly_api_cost) > 0.01:
        st.warning(f"Note: There is a small discrepancy between the Total Monthly API Cost (${total_monthly_api_cost:.5f}) and the sum of the Monthly Cost Breakdown (${total_from_pivot:.2f}). This may be due to rounding differences.")

# Tab 4: Export
with tab4:
    st.markdown('<div class="sub-header">Export Data</div>', unsafe_allow_html=True)
    st.markdown("Export your cost analysis results for further analysis or reporting.")

    # Create DataFrames for export

    # 1. Summary DataFrame
    summary_df = pd.DataFrame({
        'Metric': [
            'Organization Size',
            'Working Days per Month',
            'Per-Seat License Cost (USD/month)',
            'Weighted Average Cost per User (USD/month)',
            'Total Monthly API Cost (USD)',
            'Total Monthly Per-Seat Cost (USD)',
            'Cost Savings (USD)',
            'Cost Savings (%)'
        ],
        'Value': [
            organization_size,
            working_days,
            per_seat_license_cost,
            weighted_avg_cost_per_user,
            total_monthly_api_cost,
            total_monthly_per_seat_cost,
            cost_savings,
            (cost_savings/total_monthly_per_seat_cost*100 if total_monthly_per_seat_cost > 0 else 0)
        ]
    })

    # 2. User Archetypes DataFrame
    archetypes_df = pd.DataFrame([
        {
            'Archetype Name': archetype['name'],
            'Percentage of Employees': archetype['percentage'],
            'Prompts per Day': archetype['prompts_per_day'],
            'Input Tokens per Prompt': archetype['input_tokens_per_prompt'],
            'Output Tokens per Prompt': archetype['output_tokens_per_prompt']
        }
        for archetype in st.session_state.user_archetypes
    ])

    # 3. AI Models DataFrame
    models_df = pd.DataFrame([
        {
            'Model Name': model['name'],
            'Input Price (USD/1M tokens)': model['input_price_per_1M'],
            'Output Price (USD/1M tokens)': model['output_price_per_1M'],
            'Usage Percentage': model['usage_percentage']
        }
        for model in st.session_state.ai_models
    ])

    # 4. Detailed Results DataFrame
    detailed_results = []
    for result in results:
        for model_cost in result['model_costs']:
            detailed_results.append({
                'Archetype': result['archetype_name'],
                'Archetype Percentage': result['percentage'],
                'Model': model_cost['model_name'],
                'Monthly Input Tokens': model_cost['input_tokens'],
                'Monthly Output Tokens': model_cost['output_tokens'],
                'Input Cost (USD)': model_cost['input_cost'],
                'Output Cost (USD)': model_cost['output_cost'],
                'Total Cost (USD)': model_cost['total_cost']
            })

    detailed_results_df = pd.DataFrame(detailed_results)

    # 5. Monthly Cost Breakdown DataFrame (with weighted costs)
    monthly_breakdown_export = []
    for result in results:
        archetype_name = result['archetype_name']
        archetype_percentage = result['percentage']
        archetype_weight = archetype_percentage / 100 * organization_size

        for model_cost in result['model_costs']:
            model_name = model_cost['model_name']
            monthly_cost = model_cost['total_cost'] * archetype_weight

            monthly_breakdown_export.append({
                'Archetype': archetype_name,
                'Model': model_name,
                'Number of Employees': archetype_weight,
                'Monthly Cost (USD)': monthly_cost
            })

    monthly_breakdown_df = pd.DataFrame(monthly_breakdown_export)

    # Create pivot table for export
    pivot_table_export = pd.pivot_table(
        monthly_breakdown_df, 
        values='Monthly Cost (USD)', 
        index=['Archetype'], 
        columns=['Model'], 
        aggfunc='sum',
        fill_value=0
    )

    # Add row totals
    pivot_table_export['Total'] = pivot_table_export.sum(axis=1)

    # Add column totals
    pivot_table_export.loc['Total'] = pivot_table_export.sum(axis=0)

    # Reset index to make 'Archetype' a regular column
    pivot_table_export = pivot_table_export.reset_index()

    # Export buttons
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Summary Data")
        st.dataframe(summary_df, use_container_width=True)
        st.markdown(get_csv_download_link(summary_df, "genai_cost_summary.csv", "Download Summary CSV"), unsafe_allow_html=True)

        st.markdown("### User Archetypes Data")
        st.dataframe(archetypes_df, use_container_width=True)
        st.markdown(get_csv_download_link(archetypes_df, "genai_archetypes.csv", "Download Archetypes CSV"), unsafe_allow_html=True)

        st.markdown("### Monthly Cost Breakdown by Archetype and Model")
        st.dataframe(monthly_breakdown_df, use_container_width=True)
        st.markdown(get_csv_download_link(monthly_breakdown_df, "genai_monthly_breakdown.csv", "Download Monthly Breakdown CSV"), unsafe_allow_html=True)

    with col2:
        st.markdown("### AI Models Data")
        st.dataframe(models_df, use_container_width=True)
        st.markdown(get_csv_download_link(models_df, "genai_models.csv", "Download Models CSV"), unsafe_allow_html=True)

        st.markdown("### Detailed Results Data")
        st.dataframe(detailed_results_df, use_container_width=True)
        st.markdown(get_csv_download_link(detailed_results_df, "genai_detailed_results.csv", "Download Detailed Results CSV"), unsafe_allow_html=True)

        st.markdown("### Monthly Cost Breakdown Pivot Table")
        st.dataframe(pivot_table_export, use_container_width=True)
        st.markdown(get_csv_download_link(pivot_table_export, "genai_monthly_breakdown_pivot.csv", "Download Monthly Breakdown Pivot CSV"), unsafe_allow_html=True)

    # Export all data in a single file
    all_data = {
        'Summary': summary_df,
        'User Archetypes': archetypes_df,
        'AI Models': models_df,
        'Detailed Results': detailed_results_df,
        'Monthly Cost Breakdown': monthly_breakdown_df,
        'Monthly Cost Breakdown Pivot': pivot_table_export
    }

    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in all_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Provide download link
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="genai_cost_analysis.xlsx">Download Complete Excel Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("AI API vs AI Team Plans Calculator | Prototype Version 1.0")
