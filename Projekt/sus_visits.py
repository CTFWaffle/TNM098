import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set renderer to open in browser
pio.renderers.default = 'browser'

# Load the data
df = pd.read_csv('location_visits.csv')

# Ensure total_spent is numeric (remove $ and commas if present)
df['total_spent'] = pd.to_numeric(df['total_spent'].replace('[\$,]', '', regex=True), errors='coerce')

# Filter for rows where visit_count == 0 and total_spent != 0
filtered = df[(df['visit_count'] == 0) & (df['total_spent'] != 0)].copy()

filtered['label'] = filtered['location'] 

# Create interactive bar plot with viridis color map
fig = px.bar(
    filtered,
    x='label',
    y='total_spent',
    color='employee',
    color_discrete_sequence=px.colors.sequential.Viridis,  # Add this line for viridis colors
    title='Cases with 0 Visits but Nonzero Total Spent',
    labels={'label': 'Location', 'total_spent': 'Total Spent'},
)

# After creating the figure with px.bar
for trace in fig.data:
    trace.visible = "legendonly"

fig.update_layout(
    xaxis_tickangle=-90,
    xaxis_tickfont_size=10,
    legend_title_text='Employee',
    margin=dict(b=150)
)

# Add Select All / Deselect All buttons
num_employees = filtered['employee'].nunique()
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

fig.show()