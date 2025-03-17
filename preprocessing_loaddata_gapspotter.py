import os
import pandas as pd
import time

start_time = time.time()
# Define paths
folder_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Half hourly Hawkes bay"
aux_file_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Hawkesbay households key attributes.csv"
output_file_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE"

# Load auxiliary file
aux_data = pd.read_csv(aux_file_path)

# Prepare a master DataFrame to store all gap information across files
all_gap_info = pd.DataFrame()

# Prepare columns for start and stop datetimes
gap_columns = []

# Process each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Ensure only CSV files are processed
        file_path = os.path.join(folder_path, file_name)
        print(f"Start processing {file_name}")
        # Load the household data
        household_data = pd.read_csv(file_path)
        link_id = '_'.join(file_name.split('_')[:2])  # Extract the household ID (e.g., "rf_28")

        # Ensure datetime is parsed and time zones are removed
        household_data['r_dateTimeHalfHour'] = pd.to_datetime(
            household_data['r_dateTimeHalfHour']
        ).dt.tz_localize(None)

        # Detect gaps in the datetime column
        sorted_datetimes = household_data['r_dateTimeHalfHour'].drop_duplicates(keep='first').sort_values()  # Sort but keep datetime index intact and remove duplicates so only 1 copy of the datetime is saved. 
        time_diffs = sorted_datetimes.diff()  # Calculate time differences between consecutive rows
       
        # Check for duplicates after processing
        if sorted_datetimes.duplicated().any():
            print("Warning: Duplicates still exist!")
        else:
            print("Duplicates successfully removed.")
        
        # Identify gaps larger than 30 minutes, these are gaps in the measurement data
        gap_indices = time_diffs[time_diffs > pd.Timedelta("31min")].index          #? 31 just to be sure, sometimes it acts glitchy
        
        # Initialize lists for start/stop times
        start_times = [sorted_datetimes.iloc[0]]  # First data point is the first start time
        stop_times = []  # End of each segment

        # Process gaps
        for gap_index in gap_indices:
            # Skip if the gap_index is invalid
            if gap_index <= 0 or gap_index >= len(sorted_datetimes):
                print(f"Skipping invalid gap index: {gap_index}")
                continue
            stop_times.append(sorted_datetimes.iloc[gap_index - 1])  # Last point before the gap
            start_times.append(sorted_datetimes.iloc[gap_index])  # First point after the gap
        
        print(f"Start times for {link_id}: {start_times}")
        print(f"Stop times for {link_id}: {stop_times}")


        # Add the final stop time (end of the data)
        stop_times.append(sorted_datetimes.iloc[-1])

        # Append gap information
        gap_data = {
            'linkID': link_id,
            'n small gaps': 0,  # Initialize counts
            'n medium gaps': 0,
            'n large gaps': 0,
            'n huge gaps': 0,
        }

        gap_lengths = []  # To store gap lengths

        # Calculate gap lengths and add start/stop times to gap_data
        for i, (start, stop) in enumerate(zip(start_times, stop_times)):
            gap_data[f'startDT{i + 1}'] = start
            gap_data[f'stopDT{i + 1}'] = stop
            if i > 0:  # Skip the first start/stop pair for gap length calculation
                gap_length = start - stop_times[i - 1]
                gap_lengths.append(gap_length)
                gap_data[f'Glength_{i}'] = gap_length  # Store gap length as Glength_{i}

        # Categorize gaps
        small_gaps = sum(gap <= pd.Timedelta(hours=6) for gap in gap_lengths)
        medium_gaps = sum(pd.Timedelta(hours=6) < gap <= pd.Timedelta(hours=48) for gap in gap_lengths)
        large_gaps = sum(pd.Timedelta(hours=48) < gap <= pd.Timedelta(days=14) for gap in gap_lengths)
        huge_gaps = sum(gap > pd.Timedelta(days=14) for gap in gap_lengths)

        # Update the counts in gap_data
        gap_data['n small gaps'] = small_gaps
        gap_data['n medium gaps'] = medium_gaps
        gap_data['n large gaps'] = large_gaps
        gap_data['n huge gaps'] = huge_gaps

        print(f"Gap data for {link_id}: {gap_data}")

        # concenate gap information to the master DataFrame (.append doesn't exist anymore in pandas)
        all_gap_info = pd.concat([all_gap_info, pd.DataFrame([gap_data])], ignore_index=True)
        
        print(f"Stop processing {file_name}")

# save to new load_data_gap_info.csv
all_gap_info.to_csv(os.path.join(output_file_path, "load_data_gap_info.csv"), index=False)  


end_time = time.time()
print(f"Script runtime: {end_time - start_time:.2f} seconds")
#// Initialize an empty DataFrame for standardized profiles
#// standardized_profiles = pd.DataFrame({'timestep': range(17520)})  # Half-hourly timesteps for a full year
        # Identify circuits to sum from the auxiliary file #! fix sorting (not just re-indexing). So the rows should actually be shuffled 
        #// if link_id not in aux_data['linkID'].values:
        #//     print(f"Warning: No aggregation key found for household {link_id}. Skipping.")
        #//     continue
#// 
        #// # Extract the circuits to sum for this household
        #// circuits_to_sum = aux_data.loc[
        #//     aux_data['linkID'] == link_id,
        #//     ['CircuitsToSum1', 'CircuitsToSum2']
        #// ].values.flatten()
#// 
        #// # Sum mean power for the specified circuits
        #// total_mean_power = household_data[household_data['circuit'].isin(circuits_to_sum)] \
        #//     .groupby('r_dateTimeHalfHour')['meanPowerW'].sum()
#// 
        #// # Align to a full year starting from January 1st
        #// start_date = total_mean_power.index.min()
        #// shifted_data = total_mean_power.copy()  # Safe handling by making a copy
#// 
        #// # Check if the start date is NOT January 1st
        #// if start_date.month != 1 or start_date.day != 1:
        #//     print(f"Shifting data for {link_id} to align with January 1st.")
#// 
        #// # Replace the index with a standardized timeline
        #// shifted_data.index = pd.date_range("2023-01-01", periods=len(shifted_data), freq="30min")
#// 
        #// # Resample to half-hourly intervals, fill missing data if needed
        #// shifted_data = shifted_data.reindex(
        #//     pd.date_range("2023-01-01", periods=17520, freq="30min"),
        #//     fill_value=0
        #// )
#// 
        #// # Add to standardized profiles
        #// standardized_profiles[f'P_{link_id}'] = shifted_data.values

#// Save the standardized profiles
#// standardized_profiles.to_csv("standardized_profiles.csv", index=False)
