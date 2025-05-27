import pandas as pd

# Load the data
df1 = pd.read_csv('location_visits.csv')

df2 = pd.read_csv('Projekt\shared_locations.csv')


#print("Location Visits Data:")
#print(df1.head())

print("Shared Locations Data:")
print(df2.head())
