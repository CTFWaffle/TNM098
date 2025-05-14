import pandas as pd

# Read gps.csv
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')

# Sort by values in the 'id' column
gps_data = gps_data.sort_values(by='id')
print(gps_data['id'].unique())

# Print amount of rows
print("Amount of rows in gps_data:", len(gps_data))

# Remove all coordinates where the id is larger than 35
gps_data = gps_data[gps_data['id'] <= 35]
print("Amount of rows in gps_data after removing id > 35:", len(gps_data))