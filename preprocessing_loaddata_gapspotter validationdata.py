import os
import pandas as pd
import time
import Opt_helpers as Ohelp
import numpy as np
from tqdm import tqdm

start_time = time.time()
# Define paths
folder_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Validation data 1 household"
output_folder_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Validation data 1 household"

# Prepare a master DataFrame to store all gap information across files
all_gap_info = pd.DataFrame()

# Prepare columns for start and stop datetimes
gap_columns = []

# Process each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith("24.xlsx"):  # Ensure only CSV files are processed
        file_path = os.path.join(folder_path, file_name)
        print(f"Start processing {file_name}")
        # Load the household data
        household_data = pd.read_excel(file_path)
        # link_id = '_'.join(file_name.split('_')[:2])  # Extract the household ID (e.g., "rf_28")

        # Ensure datetime is parsed and time zones are removed
        household_data['r_dateTimeHalfHour'] = pd.to_datetime(
            household_data['Updated Time']
        ).dt.tz_localize(None)

        # Detect gaps in the datetime column
        sorted_datetimes = household_data['r_dateTimeHalfHour'] #.drop_duplicates(keep='first').sort_values()  # Sort but keep datetime index intact and remove duplicates so only 1 copy of the datetime is saved. 
        time_diffs = sorted_datetimes.diff()  # Calculate time differences between consecutive rows
       
        # Check for duplicates after processing
        if sorted_datetimes.duplicated().any():
            print("Warning: Duplicates still exist!")
        else:
            print("Duplicates successfully removed.")
        
        # Identify gaps larger than 30 minutes, these are gaps in the measurement data
        gap_indices = time_diffs[time_diffs > pd.Timedelta("6min")].index          #? 31 just to be sure, sometimes it acts glitchy
        print("gap indices: ")
        print(gap_indices)
        
        gap_indices_list = gap_indices.tolist()
        Ohelp.makeJSON(gap_indices_list, "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Validation data 1 household/gap_indices.json")
        
        
        
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
        
        # print(f"Start times: {start_times}")
        # print(f"Stop times: {stop_times}")


        # Add the final stop time (end of the data)
        stop_times.append(sorted_datetimes.iloc[-1])

        # Append gap information
        gap_data = {
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

        #print(f"Gap data: {gap_data}")

        # concenate gap information to the master DataFrame (.append doesn't exist anymore in pandas)
        all_gap_info = pd.concat([all_gap_info, pd.DataFrame([gap_data])], ignore_index=True)
        
        print(f"Stop processing {file_name}")


# save to new load_data_gap_info.csv
all_gap_info.to_csv(os.path.join(output_folder_path, "validation_data_gap_info.csv"), index=False)  


#* HELPER FUNCTIONS
def interpolate_gap_columns(stop_time, start_time, gap_range, matching_rows, copy_household_data):
    """
    Linearly interpolate values for specific columns between stop_time and start_time
    for each unique circuit.
    """
    gap_rows = []

    for _, stop_row in matching_rows.iterrows():
        # Ensure we match the correct circuit for start_row
        start_row = copy_household_data.loc[
            (copy_household_data['r_dateTimeHalfHour'] == start_time)
        ]

        if start_row.empty:
            print(f"Warning: Missing start data for at {start_time}. Skipping.")
            continue  # Skip interpolation if start data is missing for this circuit




        # Extract values for interpolation
        stop_values = stop_row[['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W', 'SoC_perc']]
        start_values = start_row[['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W', 'SoC_perc']].iloc[0]

        # Perform linear interpolation for each time point in the gap range
        for time_point in gap_range:
            # Linear interpolation formula
            weight = (time_point - stop_time) / (start_time - stop_time)
            interpolated_values = stop_values + weight * (start_values - stop_values)
            
            # Create a new row with interpolated values
            gap_row = stop_row.copy()
            gap_row['r_dateTimeHalfHour'] = time_point
            gap_row[['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W', 'SoC_perc']] = interpolated_values
            gap_rows.append(gap_row)

    return gap_rows

def copy_values_from_earlier_sequence_optimized(stop_time, gap_length, gap_range, matching_rows, original_data):
    """
    Fills gaps by copying values from an earlier sequence of rows corresponding to the same time range.
    Optimized to first identify valid times using one circuit and then perform batch copying for all circuits.
    """
    copied_rows = []

    # Reference circuit for checking data availability

    # Find valid rows for the reference circuit
    valid_rows = original_data.loc[
        (original_data['r_dateTimeHalfHour'] < stop_time) &  # Only look at rows before the gap
        (original_data['r_dateTimeHalfHour'] > stop_time - pd.Timedelta(days=6))  # Limit to 6 days back
    ]

    valid_times = []

    for gap_time in gap_range:
        # Find the most recent sequence of rows matching the gap time
        if not valid_rows.empty:
                sequence_start = max(gap_time - pd.Timedelta(days=(gap_length.days + 1)), pd.Timestamp.min)
                sequence_rows = valid_rows.loc[
                    (valid_rows['r_dateTimeHalfHour'] == sequence_start) &
                    (valid_rows['r_dateTimeHalfHour'] < gap_time)  # Exclude the gap itself
                ]
                if not sequence_rows.empty:
                    valid_times.append(sequence_rows.iloc[0]['r_dateTimeHalfHour'])

    # Check if valid times cover the entire gap range
    #print(f"Valid times = {valid_times}")
    
    if len(valid_times) < len(gap_range):
        print(f"Warning: Unable to fill the gap from {gap_range[0]} to {gap_range[-1]}. Proceeding with method for longer gaps")
        return copied_rows

    # Perform batch copying for all circuits
    for _, template_row in matching_rows.iterrows():

        # Filter valid rows for this circuit
        valid_rows_for_circuit = original_data.loc[
            (original_data['r_dateTimeHalfHour'].isin(valid_times))
        ]

        # Create a DataFrame that matches the template_row structure
        mass_copied = pd.DataFrame([template_row.to_dict()] * len(gap_range))
        mass_copied['r_dateTimeHalfHour'] = gap_range  # Update timestamps

        # Copy power columns from valid_rows_for_circuit into the mass_copied DataFrame
        power_columns = ['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W', 'SoC_perc']
        for column in power_columns:
            mass_copied[column] = valid_rows_for_circuit[column].values  # Assign column values directly

        # Append the mass_copied rows to the copied_rows list
        copied_rows.extend(mass_copied.to_dict(orient='records'))

    return copied_rows

def copy_values_from_nearby_weeks_fallback(gap_range, matching_rows, original_data):
    
    """
    Fills gaps by copying values from nearby weeks for all circuits, using prevalidation with sampling and rigorous checking.
    Optimized to first identify valid times using one circuit and then perform batch copying for all circuits.
    """
    copied_rows = []

    # Reference circuit for checking data availability
    week_offsets = [-1, +1, -2, +2, -3, +3, -4, +4, -5, +5, -6, +6, -7, +7, -8, +8, -9, +9, -10, +10, -11, +11, -12, +12]

    # Sampled gap times for quick prevalidation
    sampled_gap_times = [gap_range[0], gap_range[-1]] + [
        gap_range[i] for i in range(0, len(gap_range), max(1, len(gap_range) // 100))
    ]

    #* Prevalidation: Check sampled times across week offsets
    for week_offset in tqdm(week_offsets, desc="Checking sampled weeks, breaks if succesful"):
        found_all_samples = True
        for sample_time in sampled_gap_times:
            candidate_time = sample_time + pd.Timedelta(weeks=week_offset)
            candidate_rows = original_data.loc[
                (original_data['r_dateTimeHalfHour'] == candidate_time) 
            ]
            if candidate_rows.empty:
                found_all_samples = False
                break  # Stop checking this week offset if any sample is invalid

        if found_all_samples:
            print(f"Valid sampled times found for week offset {week_offset}. Proceeding with rigorous checking.")
            # Proceed to rigorous checking
            valid_times = []
            for gap_time in tqdm(gap_range, desc=f"Checking all times for week offset {week_offset}"):
                candidate_time = gap_time + pd.Timedelta(weeks=week_offset)
                candidate_rows = original_data.loc[
                    (original_data['r_dateTimeHalfHour'] == candidate_time)
                ]
                if not candidate_rows.empty:
                    valid_times.append(candidate_time)

            if len(set(valid_times)) == len(gap_range):
                print(f"Found valid rows for all gap times using week offset {week_offset}. Proceeding to copy data.")
                break

    if len(set(valid_times)) < len(gap_range):
        print(f"Warning: Unable to fill the gap from {gap_range[0]} to {gap_range[-1]} using nearby weeks.")
        return copied_rows

    # Perform batch copying for all circuits
    for _, template_row in matching_rows.iterrows():

        # Filter valid rows for this circuit
        valid_rows_for_circuit = original_data.loc[
            (original_data['r_dateTimeHalfHour'].isin(valid_times))
        ]

        # Create a DataFrame that matches the template_row structure
        mass_copied = pd.DataFrame([template_row.to_dict()] * len(gap_range))
        mass_copied['r_dateTimeHalfHour'] = gap_range  # Update timestamps

        # Copy power columns from valid_rows_for_circuit into the mass_copied DataFrame
        power_columns = ['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W', 'SoC_perc']
        for column in power_columns:
            mass_copied[column] = valid_rows_for_circuit[column].values  # Assign column values directly

        # Append the mass_copied rows to the copied_rows list
        copied_rows.extend(mass_copied.to_dict(orient='records'))

    return copied_rows

#* ACTUAL FILLING

# Define paths and load gap data
# key_file_path = r"C:\Users\20193915\OneDrive - TU Eindhoven\UoA-INTERNSHIP\UoA_CODE\validation_data_gap_info.csv"

# gap_key = pd.read_csv(key_file_path)

gap_key = all_gap_info

# Process each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith("24.xlsx"):  # Ensure only CSV files are processed
        file_path = os.path.join(folder_path, file_name)
        print(f"Start processing FOR FILLING {file_name}")
        
        # Load the household data
        og_household_data = pd.read_excel(file_path)
        #link_id = '_'.join(file_name.split('_')[:2])  # Extract the household ID (e.g., "rf_28")
        
        # Create a copy
        copy_household_data = og_household_data.copy()
        
        # Ensure datetime is parsed and time zones are removed
        copy_household_data['r_dateTimeHalfHour'] = pd.to_datetime(
            copy_household_data['Updated Time']
        ).dt.tz_localize(None)
        copy_household_data.drop(columns="Time Zone")
        
        # Retrieve gap data for the current household
        key_gap_data = gap_key.iloc[:, 5:].replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False).dropna(axis=1)
        key_gap_dict = key_gap_data.to_dict(orient='records')[0]
        
        
        print(f"Processing gaps: {key_gap_dict}")
                
        num_gaps = sum(1 for key in key_gap_dict if key.startswith('Glength_'))
        print(f"Number of gaps: {num_gaps}")
        
        # Prepare for filling gaps
        filled_data = []
        for i in range(1, num_gaps + 1):  # Assuming 3 entries per gap (stopDT, startDT, Glength)
            stop_key = f"stopDT{i}"
            start_key = f"startDT{i + 1}"
            gap_length_key = f"Glength_{i}"

            if stop_key in key_gap_dict and start_key in key_gap_dict:
                stop_time = pd.to_datetime(key_gap_dict[stop_key], dayfirst=True)
                start_time = pd.to_datetime(key_gap_dict[start_key], dayfirst=True)
                gap_length = pd.Timedelta(key_gap_dict[gap_length_key])
                print(f"Current key_gap_dict entry for {gap_length_key}: {key_gap_dict[gap_length_key]}")

                # Find all rows with this stop_time
                matching_rows = copy_household_data.loc[
                    copy_household_data['r_dateTimeHalfHour'] == stop_time
                ]

                gap_range = pd.date_range(
                    stop_time + pd.Timedelta("5min"),
                    start_time - pd.Timedelta("5min"),
                    freq="5min"
                )
                #// print(f"gap_range={gap_range}")

                if gap_length <= pd.Timedelta(hours=6):
                    interpolated_rows = interpolate_gap_columns(stop_time, start_time, gap_range, matching_rows, copy_household_data)
                    filled_data.extend(interpolated_rows)
                    print(f"Added interpolated rows to filled data")
                elif gap_length <= pd.Timedelta(days=2):
                    print("found a gap under 48 hours")
                    copied_rows = copy_values_from_earlier_sequence_optimized(stop_time, gap_length, gap_range, matching_rows, copy_household_data)
                    if copied_rows:  # Check if the earlier-years method successfully filled the gap
                                print(f"Succes: copied the values from a nearby sequence")
                                print (f"First in copied rows: {copied_rows[0]}")
                                filled_data.extend(copied_rows)
                    else:
                        copied_rows = copy_values_from_nearby_weeks_fallback(gap_range, matching_rows, copy_household_data)
                        if copied_rows:  # Only extend if valid rows were filled
                            print(f"Succes: copied the values from nearby weeks")
                            print (f"First in copied rows: {copied_rows[0]}")
                            filled_data.extend(copied_rows)
                        else:
                            print(f"Warning: Gap starting at {stop_time} could not be filled.")

                    print(f"Added copied rows from nearby hours/days to filled data")


        
        #? Debugging prints
        # print(f"Sample of filled_data: {filled_data[:5]}")
        # print(f"Type of elements in filled_data: {[type(row) for row in filled_data[:5]]}")
        
        # Add original rows and sort
        filled_data = [row.to_dict() if isinstance(row, pd.Series) else row for row in filled_data]             #? Apparently this is necessary. Don't understand why but shout out to chatgpt
        copy_household_data = pd.concat([copy_household_data, pd.DataFrame(filled_data)], ignore_index=True)
        copy_household_data = copy_household_data.sort_values(by='r_dateTimeHalfHour').reset_index(drop=True)
        copy_household_data.drop(columns=["Updated Time", "Time Zone"], inplace=True)
        copy_household_data.insert(0, "r_dateTimeHalfHour", copy_household_data.pop("r_dateTimeHalfHour"))

        #* resampling to make 30 min bins
        copy_household_data = copy_household_data.copy()
        copy_household_data.set_index('r_dateTimeHalfHour', inplace=True)

        # Shift timestamps back by 25 minutes to align with the intended bins

        copy_household_data = copy_household_data.resample('30min', label='left', closed='left').mean()

        # Drop the last row if it is incomplete (optional)
        #copy_household_data.drop(copy_household_data.index[-1], axis=0, inplace=True)

        copy_household_data.reset_index(inplace=True)
        
        copy_household_data = copy_household_data[~((copy_household_data['r_dateTimeHalfHour'].dt.day == 29) & (copy_household_data['r_dateTimeHalfHour'].dt.month == 2))].reset_index(drop=True)
        copy_household_data['Production_Power_kW'] = copy_household_data['Production_Power_W']/1000
        copy_household_data['Consumption_Power_kW'] = copy_household_data['Consumption_Power_W']/1000
        copy_household_data['Grid_Power_kW'] = copy_household_data['Grid_Power_W']/1000
        copy_household_data['Purchasing_Power_kW'] = copy_household_data['Purchasing_Power_W']/1000
        copy_household_data['Feedin_Power_kW'] = copy_household_data['Feedin_Power_W']/1000
        copy_household_data['Battery_Power_kW'] = copy_household_data['Battery_Power_W']/1000
        copy_household_data['Charging_Power_kW'] = copy_household_data['Charging_Power_W']/1000
        copy_household_data['Discharging_Power_kW'] = copy_household_data['Discharging_Power_W']/1000
        
        copy_household_data.drop(columns=['Production_Power_W', 'Consumption_Power_W', 'Grid_Power_W', 'Purchasing_Power_W', 'Feedin_Power_W', 'Battery_Power_W', 'Charging_Power_W', 'Discharging_Power_W'], inplace=True)
        
        # Save the processed data to a new CSV file
        output_file_name = f"2024validation_prepared_data.csv"
        output_file_path = os.path.join(output_folder_path, output_file_name)
        copy_household_data.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")




end_time = time.time()

