import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from pyproj import CRS
from pykml import parser
import seaborn as sns

# Load the datasets
credit_card_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
loyalty_card_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')

# Get unique locations from datasets
#unique_locations = credit_card_data['location'].unique()
#unique_locations_loyalty = loyalty_card_data['location'].unique()

# Get unique locations from both datasets
#unique_locations_combined = set(unique_locations).union(set(unique_locations_loyalty))


# Mapping between the coordinates of places and names
places = {
    'Hallowed Grounds': (0.0,0.0), 
    "Guy's Gyros": (0.0,0.0), 
    'Desafio Golf Course': (0.0,0.0),
    'Bean There Done That': (0.0,0.0), 
    "Frydos Autosupply n' More": (0.0,0.0), 
    "Frank's Fuel": (0.0,0.0), 
    'Daily Dealz': (0.0,0.0), 
    "Jack's Magical Beans": (0.0,0.0), 
    'Nationwide Refinery': (0.0,0.0), 
    'Abila Zacharo': (0.0,0.0), 
    'Kronos Pipe and Irrigation': (0.0,0.0), 
    'Carlyle Chemical Inc.': (0.0,0.0), 
    "Octavio's Office Supplies": (0.0,0.0),
    'Hippokampos': (0.0,0.0), 
    'Abila Scrapyard': (0.0,0.0), 
    'General Grocer': (0.0,0.0), 
    'Abila Airport': (0.0,0.0), 
    'Kronos Mart': (0.0,0.0), 
    'Chostus Hotel': (0.0,0.0), 
    'U-Pump': (0.0,0.0), 
    "Brew've Been Served": (0.0,0.0), 
    'Maximum Iron and Steel': (0.0,0.0), 
    'Roberts and Sons': (0.0,0.0), 
    'Coffee Shack': (0.0,0.0), 
    'Stewart and Sons Fabrication': (0.0,0.0), 
    'Ahaggo Museum': (0.0,0.0), 
    'Katerina’s Café': (0.0,0.0), 
    'Gelatogalore': (0.0,0.0), 
    'Kalami Kafenion': (0.0,0.0), 
    'Brewed Awakenings': (0.0,0.0), 
    'Ouzeri Elian': (0.0,0.0), 
    "Albert's Fine Clothing": (0.0,0.0), 
    "Shoppers' Delight": (0.0,0.0), 
    'Coffee Cameleon': (0.0,0.0)
}

# Check how many times each place appears in the datasets
def count_places(data, places):
    place_counts = {place: 0 for place in places}
    for place in places:
        place_counts[place] = data[data['location'] == place].shape[0]
    return place_counts

# Count the occurrences of each place in the credit card data
credit_card_counts = count_places(credit_card_data, places)

# Count the occurrences of each place in the loyalty card data
loyalty_card_counts = count_places(loyalty_card_data, places)

# Histogram for both datasets
plt.figure(figsize=(12, 6))
plt.bar(credit_card_counts.keys(), credit_card_counts.values(), label='Credit Card Data', alpha=0.5)
plt.bar(loyalty_card_counts.keys(), loyalty_card_counts.values(), label='Loyalty Card Data', alpha=0.5)
plt.xlabel('Places')
plt.ylabel('Number of Occurrences')
plt.title('Occurrences of Places in Credit Card and Loyalty Card Data')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Timestamps for the transactions
credit_card_data['timestamp'] = pd.to_datetime(credit_card_data['timestamp'])
loyalty_card_data['timestamp'] = pd.to_datetime(loyalty_card_data['timestamp'])

# Extract the date from the timestamp
credit_card_data['date'] = credit_card_data['timestamp'].dt.date
loyalty_card_data['date'] = loyalty_card_data['timestamp'].dt.date
# Count the number of transactions per day
credit_card_daily_counts = credit_card_data.groupby('date').size()
loyalty_card_daily_counts = loyalty_card_data.groupby('date').size()
# Plot the daily counts
plt.figure(figsize=(12, 6))
plt.plot(credit_card_daily_counts.index, credit_card_daily_counts.values, label='Credit Card Data', alpha=0.5)
plt.plot(loyalty_card_daily_counts.index, loyalty_card_daily_counts.values, label='Loyalty Card Data', alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.title('Daily Transactions in Credit Card and Loyalty Card Data')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Group data by date and location, then count transactions
credit_card_pivot = pd.crosstab(credit_card_data['date'], credit_card_data['location'])
loyalty_card_pivot = pd.crosstab(loyalty_card_data['date'], loyalty_card_data['location'])

# Create heatmaps
fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# Credit card heatmap with smaller fonts
sns.heatmap(credit_card_pivot, cmap="YlGnBu", ax=axes[0], cbar_kws={'label': 'Number of Transactions'})

# Loyalty card heatmap with smaller fonts
sns.heatmap(loyalty_card_pivot, cmap="YlOrRd", ax=axes[1], cbar_kws={'label': 'Number of Transactions'})

plt.tight_layout()
plt.show()

# Create a difference pivot table with ABSOLUTE differences
# First fix the calculation to properly subtract loyalty card data by date and location
difference_pivot = credit_card_pivot.subtract(loyalty_card_pivot, fill_value=0)

# Take the absolute value to focus on magnitude of difference, not direction
abs_difference_pivot = difference_pivot.abs()

# Create a single heatmap showing the differences
fig, ax = plt.subplots(figsize=(16, 10))

# Use a grayscale colormap - 'Greys' or 'gray_r' 
# 'gray_r' is reversed grayscale where darker colors indicate higher values
cmap = plt.cm.gray_r  # Dark colors for high values

# Create heatmap showing the magnitude of differences
sns.heatmap(abs_difference_pivot, cmap=cmap, ax=ax, 
            cbar_kws={'label': 'Absolute Difference in Number of Transactions'})

ax.set_title('Magnitude of Difference Between Credit Card and Loyalty Card Transactions', fontsize=12)
ax.set_ylabel('Date', fontsize=10)
ax.set_xlabel('Places', fontsize=10)
ax.tick_params(axis='both', labelsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, ha='right')

plt.tight_layout()
plt.show()

# 2014 January 11,12,18,19 were weekends (maybe something happened)
# Katerinas cafe is the only one open on weekends, sus -_-
# 18 January 2014 is the largest difference in transactions between loyalty and credit cards, which happened at Katerina's Cafe, sus -_-