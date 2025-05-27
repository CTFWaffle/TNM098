import pandas as pd
import plotly.express as px
import plotly.io as pio

df = pd.read_csv('employee_summary.csv')

# Set renderer to open in browser
pio.renderers.default = 'browser'

# Create a horizontal bar chart (note the swapped x and y parameters)
fig = px.bar(
    df,
    x='Employee',  # Employee names now on y-axis
    y='Total Spent',  # Total spent on x-axis
    title='Total Spent by Employee',
    labels={'Employee': 'Employee', 'Total Spent': 'Total Spent ($)'},
    height=800,
    color_discrete_sequence=['#3366CC'],
)

# Format the layout for better readability
fig.update_layout(
    yaxis_title='Employee',  # y-axis is now Employee
    xaxis_title='Total Spent ($)',  # x-axis is now Total Spent
    margin=dict(l=150),
    autosize=True,
    xaxis=dict(
        categoryorder='total ascending',  # This ensures bars are ordered by their values
    ),
    yaxis=dict(
        type='linear',  # Ensure linear numerical axis
        autorange=True,  # Allow automatic range calculation
        tickformat='$,.2f',  # Format ticks as currency
    )
)

fig.show()