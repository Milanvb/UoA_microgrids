import os
import pandas as pd
import numpy as np
import time
#todo run them for all .csv

# Import the helper functions
from gap_filling_helpers import interpolate_gap_columns, copy_values_from_earlier_sequence_optimized, copy_values_from_earlier_years_optimized, copy_values_from_nearby_weeks_fallback

start_runtime = time.time()

# Define paths and load gap data
folder_path = "C:/Users/20193915/OneDrive - TU Eindhoven/UoA-INTERNSHIP/UoA_CODE/Half hourly Hawkes bay"
key_file_path = r"C:\Users\20193915\OneDrive - TU Eindhoven\UoA-INTERNSHIP\UoA_CODE\load_data_gap_info.csv"
output_folder_path = r"C:\Users\20193915\OneDrive - TU Eindhoven\UoA-INTERNSHIP\UoA_CODE\Half hourly Hawkes bay\Processed_load_data_complete_dataset"

gap_key = pd.read_csv(key_file_path)


# Process each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.startswith("rf_47"):  # Ensure only CSV files are processed
        file_path = os.path.join(folder_path, file_name)
        print(f"Start processing {file_name}")
        
        # Load the household data
        og_household_data = pd.read_csv(file_path)
        link_id = '_'.join(file_name.split('_')[:2])  # Extract the household ID (e.g., "rf_28")
        
        # Create a copy
        copy_household_data = og_household_data.copy()
        
        # Ensure datetime is parsed and time zones are removed
        copy_household_data['r_dateTimeHalfHour'] = pd.to_datetime(
            copy_household_data['r_dateTimeHalfHour']
        ).dt.tz_localize(None)
        
        # Retrieve gap data for the current household
        keyrow = gap_key.loc[gap_key['linkID'] == link_id]
        key_gap_data = keyrow.iloc[:, 8:].replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False).dropna(axis=1)
        key_gap_dict = key_gap_data.to_dict(orient='records')[0]
        
        
        print(f"Processing gaps for {link_id}: {key_gap_dict}")
                
        num_gaps = sum(1 for key in key_gap_dict if key.startswith('Glength_'))
        print(f"Number of gaps for {link_id}: {num_gaps}")
        
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
                    stop_time + pd.Timedelta("30min"),
                    start_time - pd.Timedelta("30min"),
                    freq="30min"
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
                elif gap_length > pd.Timedelta(days=2):                                 #todo NEED TO FIX fallback option (search nearby weeks). 
                            print("found a gap over 48 hours")
                            copied_rows = copy_values_from_earlier_years_optimized(
                                gap_range, matching_rows, copy_household_data
                            )

                            if copied_rows:  # Check if the earlier-years method successfully filled the gap
                                print(f"Succes: copied the values from the same dates in previous years")
                                print (f"First in copied rows: {copied_rows[0]}")
                                filled_data.extend(copied_rows)
                            else:
                                print(f"No valid data found in previous years for gap starting at {stop_time}, now trying to find one in nearby weeks.")
                                # Try nearby weeks as a fallback
                                copied_rows = copy_values_from_nearby_weeks_fallback(gap_range, matching_rows, copy_household_data)
                                if copied_rows:  # Only extend if valid rows were filled
                                    print(f"Succes: copied the values from nearby weeks")
                                    print (f"First in copied rows: {copied_rows[0]}")
                                    filled_data.extend(copied_rows)
                                else:
                                    print(f"Warning: Gap starting at {stop_time} could not be filled.")

        
        #? Debugging prints
        # print(f"Sample of filled_data: {filled_data[:5]}")
        # print(f"Type of elements in filled_data: {[type(row) for row in filled_data[:5]]}")
        
        # Add original rows and sort
        filled_data = [row.to_dict() if isinstance(row, pd.Series) else row for row in filled_data]             #? Apparently this is necessary. Don't understand why but shout out to chatgpt
        copy_household_data = pd.concat([copy_household_data, pd.DataFrame(filled_data)], ignore_index=True)
        copy_household_data = copy_household_data.sort_values(by='r_dateTimeHalfHour').reset_index(drop=True)
        
        # Save the processed data to a new CSV file
        output_file_name = f"{link_id}_prepared_data.csv"
        output_file_path = os.path.join(output_folder_path, output_file_name)
        copy_household_data.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

end_runtime = time.time()
print(f"Script runtime: {end_runtime - start_runtime:.2f} seconds")
