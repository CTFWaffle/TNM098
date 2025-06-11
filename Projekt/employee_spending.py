import pandas as pd
import plotly.express as px
import plotly.io as pio

# Load employee spending summary data from CSV
df = pd.read_csv('employee_summary.csv')

# Set Plotly to render plots in the default web browser
pio.renderers.default = 'browser'

# Create a stacked bar chart to visualize employee spending by payment method
fig = px.bar(
    df,
    x='Employee',  # Employee names on the x-axis
    y=['Loyalty Card Spent', 'Credit Card Spent'],  # Amounts spent by payment method
    title='Employee Spending by Payment Method',
    labels={
        'Employee': 'Employee',
        'value': 'Amount Spent ($)',
        'variable': 'Payment Method'
    },
    height=800,  # Chart height in pixels
    color_discrete_sequence=['#FF9900', '#3366CC'],  # Custom colors for each payment method
    barmode='stack'  # Stack bars to show total and breakdown by method
)

# Update chart layout for improved readability and appearance
fig.update_layout(
    yaxis_title='Amount Spent ($)',  # Label for y-axis
    xaxis_title='Employee',          # Label for x-axis
    margin=dict(l=150),              # Add left margin for long employee names
    autosize=True,                   # Enable automatic sizing
    xaxis=dict(
        categoryorder='total ascending',  # Order employees by total spending (ascending)
    ),
    yaxis=dict(
        type='linear',
        autorange=True,
        tickformat='$,.2f',  # Format y-axis ticks as currency
    ),
    legend_title_text='Payment Method'   # Title for the legend
)

# Display the interactive chart in the browser
fig.show()