import pandas as pd
import plotly.express as px
import plotly.io as pio

# =========================
# Plotly Renderer Setup
# =========================

# Set Plotly to open plots in the default web browser
pio.renderers.default = 'browser'

# =========================
# Data Loading & Cleaning
# =========================

# Load the location visits data
df = pd.read_csv('location_visits.csv')

# Ensure 'total_spent' is numeric (remove $ and commas if present)
df['total_spent'] = pd.to_numeric(
    df['total_spent'].replace('[\$,]', '', regex=True),
    errors='coerce'
)

# =========================
# Data Filtering
# =========================

# Filter for cases where there are 0 visits but nonzero spending
filtered = df[(df['visit_count'] == 0) & (df['total_spent'] != 0)].copy()

# Create a label column for plotting (can be customized as needed)
filtered['label'] = filtered['location']

# =========================
# Interactive Bar Plot
# =========================

# Create an interactive bar plot using Plotly Express with Viridis color map
fig = px.bar(
    filtered,
    x='label',
    y='total_spent',
    color='employee',
    color_discrete_sequence=px.colors.sequential.Viridis,
    title='Cases with 0 Visits but Nonzero Total Spent',
    labels={'label': 'Location', 'total_spent': 'Total Spent'},
)

# Hide all traces by default (user can select from legend)
for trace in fig.data:
    trace.visible = "legendonly"

# Update layout for better readability
fig.update_layout(
    xaxis_tickangle=-90,
    xaxis_tickfont_size=10,
    legend_title_text='Employee',
    margin=dict(b=150)  # Add space for long x-axis labels
)

# =========================
# Add Select All / Deselect All Buttons
# =========================

# Number of unique employees (for controlling trace visibility)
num_employees = filtered['employee'].nunique()

# Add interactive buttons to select/deselect all employees
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=1.15,
            showactive=True,
            buttons=[
                dict(
                    label="Select All",
                    method="update",
                    args=[{"visible": [True] * num_employees}]
                ),
                dict(
                    label="Deselect All",
                    method="update",
                    args=[{"visible": ["legendonly"] * num_employees}]
                ),
            ],
        )
    ]
)

# =========================
# Show Plot
# =========================

fig.show()