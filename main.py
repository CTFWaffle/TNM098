import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read Eyetracking data
eyetracking_data = pd.read_table('EyeTrack-raw.tsv')
print(eyetracking_data.head())

# Convert the 'Recording Timestamp' column to seconds
