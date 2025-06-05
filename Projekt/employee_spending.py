import pandas as pd
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('employee_summary.csv')

print(df)

# Set renderer to open in browser
pio.renderers.default = 'browser'

# Create a stacked bar chart showing both spending types
fig = px.bar(
    df,
    x='Employee',
    y=['Loyalty Card Spent', 'Credit Card Spent'],  # Use sum columns instead of transaction counts
    title='Employee Spending by Payment Method',
    labels={
        'Employee': 'Employee', 
        'value': 'Amount Spent ($)', 
        'variable': 'Payment Method'
    },
    height=800,
    color_discrete_sequence=['#FF9900','#3366CC'],  # Different colors for categories
    barmode='stack'  # Stack the bars to show total and components
)

# Format the layout for better readability
fig.update_layout(
    yaxis_title='Amount Spent ($)',
    xaxis_title='Employee',
    margin=dict(l=150),
    autosize=True,
    xaxis=dict(
        categoryorder='total ascending',  # Order by total amount
    ),
    yaxis=dict(
        type='linear',
        autorange=True,
        tickformat='$,.2f',  # Format ticks as currency
    ),
    legend_title_text='Payment Method'
)

fig.show()