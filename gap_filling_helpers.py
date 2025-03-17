# gap_filling_helpers
import pandas as pd
from tqdm import tqdm
import numpy as np

def interpolate_gap_columns(stop_time, start_time, gap_range, matching_rows, copy_household_data):
    """
    Linearly interpolate values for specific columns between stop_time and start_time
    for each unique circuit.
    """
    gap_rows = []

    for _, stop_row in matching_rows.iterrows():
        # Ensure we match the correct circuit for start_row
        circuit = stop_row['circuit']
        start_row = copy_household_data.loc[
            (copy_household_data['r_dateTimeHalfHour'] == start_time) & 
            (copy_household_data['circuit'] == circuit)
        ]

        if start_row.empty:
            print(f"Warning: Missing start data for circuit '{circuit}' at {start_time}. Skipping.")
            continue  # Skip interpolation if start data is missing for this circuit

        # Extract values for interpolation
        stop_values = stop_row[['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']]
        start_values = start_row[['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']].iloc[0]

        # Perform linear interpolation for each time point in the gap range
        for time_point in gap_range:
            # Linear interpolation formula
            weight = (time_point - stop_time) / (start_time - stop_time)
            interpolated_values = stop_values + weight * (start_values - stop_values)
            
            # Create a new row with interpolated values
            gap_row = stop_row.copy()
            gap_row['r_dateTimeHalfHour'] = time_point
            gap_row[['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']] = interpolated_values
            gap_row['nObs'] = 0  # Fill nObs with 0 for gap rows
            gap_rows.append(gap_row)

    return gap_rows


def copy_values_from_earlier_sequence_optimized(stop_time, gap_length, gap_range, matching_rows, original_data):
    """
    Fills gaps by copying values from an earlier sequence of rows corresponding to the same time range.
    Optimized to first identify valid times using one circuit and then perform batch copying for all circuits.
    """
    copied_rows = []

    # Reference circuit for checking data availability
    first_circuit = matching_rows.iloc[0]['circuit']

    # Find valid rows for the reference circuit
    valid_rows = original_data.loc[
        (original_data['circuit'] == first_circuit) &
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
        circuit = template_row['circuit']

        # Filter valid rows for this circuit
        valid_rows_for_circuit = original_data.loc[
            (original_data['r_dateTimeHalfHour'].isin(valid_times)) &
            (original_data['circuit'] == circuit)
        ]

        # Create a DataFrame that matches the template_row structure
        mass_copied = pd.DataFrame([template_row.to_dict()] * len(gap_range))
        mass_copied['r_dateTimeHalfHour'] = gap_range  # Update timestamps
        mass_copied['nObs'] = 0  # Update nObs for gap rows

        # Copy power columns from valid_rows_for_circuit into the mass_copied DataFrame
        power_columns = ['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']
        for column in power_columns:
            mass_copied[column] = valid_rows_for_circuit[column].values  # Assign column values directly

        # Append the mass_copied rows to the copied_rows list
        copied_rows.extend(mass_copied.to_dict(orient='records'))

    return copied_rows


def copy_values_from_earlier_years_optimized(gap_range, matching_rows, original_data):
    """
    Fills gaps by copying values from earlier or later years for all circuits,
    using a pre-searching strategy for the first and last gap times to validate feasibility.
    """
    copied_rows = []

    # Reference circuit for checking data availability
    first_circuit = matching_rows.iloc[0]['circuit']
    years_offset = [-1, +1, -2, +2, -3, +3]  # Max range: ±3 years

    # Sample only the first and last gap times for pre-checking
    sampled_gap_times = [gap_range[0], gap_range[-1]]

    #* Pre-search for valid years
    for year_offset in tqdm(years_offset, desc=f'Checking nearby years, will break if successful'):  # Check per yearoffset the sample times,  
        print(f" Checking offset: {year_offset}")
        
        # Make variables for checking candidates. Has to be cumulative since all the candidates must be valid
        cum_candidate_rows = []
        cum_candidate_rows_rigorous = []
                                               
        for sample_time in sampled_gap_times:                                                       # so cycles through year -1 first, with stop_time then start_time, then +1
            candidate_time = sample_time + pd.DateOffset(years=year_offset)
            candidate_rows = original_data.loc[                                                     # Stores all the rows that exist
                (original_data['r_dateTimeHalfHour'] == candidate_time) &                           # With stop_time and then start_time, for year -1 first etc. etc. 
                (original_data['circuit'] == first_circuit)                                         # Only stores 1 circuit, to reduce computing time later on. 
            ]
            
            cum_candidate_rows.append(candidate_rows)
            
        
        if not len(cum_candidate_rows) == len(sampled_gap_times):                
            print(f" The year offset: {year_offset}, ending with start_time to fill the gap: {candidate_time} was NOT valid. Proceeding to test the next closest year")
            
        if len(cum_candidate_rows) == len(sampled_gap_times):       # Should contain two, otherwise the year is not valid. #? Technically don't need this if statement, but it makes the code readible
            print(f" The year offset: {year_offset}, ending with start_time to fill the gap: {candidate_time} IS VALID. Proceeding with rigorous testing...")
                
            #* Proceed to check rigorously (i.e. whether there are all the datetimes to fill the complete gap) #! Won't be reached if the pre-check is not valid
            for gap_time in tqdm(gap_range, desc=f"Checking whether the full gap_range can be filled with year offset {year_offset}"):      # cycles over all the times in the gap, still only in for the first circuit
                candidate_time_rigorous = gap_time + pd.DateOffset(years=year_offset)
                candidate_rows_rigorous = original_data.loc[                            # Stores all valid rows that could fill the gap, using the first valuable year_offset found
                    (original_data['r_dateTimeHalfHour'] == candidate_time_rigorous) &
                    (original_data['circuit'] == first_circuit)
                ]
                if candidate_rows_rigorous.empty:
                    break  # Stop checking this week offset if any sample is invalid
                else:                
                    cum_candidate_rows_rigorous.append(candidate_rows_rigorous)
                
            if len(cum_candidate_rows_rigorous) == len(gap_range):      # if this condition is not met, it should go back to quick checking the next year. 
                print(f"Found {len(cum_candidate_rows_rigorous)} valid rows to fill a gap with {len(gap_range)}")
                #? Debugprint                
                print("Rigorous candidate row, first entry in list:", cum_candidate_rows_rigorous [0])
                print("Rigorous candidate row, last entry in list:", cum_candidate_rows_rigorous [-1])
                # Combine all candidate rows into a single DataFrame
                valid_times_df = pd.concat(cum_candidate_rows_rigorous, ignore_index=True)
                
                #? Debugprint                
                #print("Checking valid_times_df:", valid_times_df)                
                valid_times = valid_times_df['r_dateTimeHalfHour'].unique()
                
                #? Debugprints
                print(f"valid times: {valid_times}")
                

                #* Batch Copying for All Circuits
                for _, template_row in tqdm(matching_rows.iterrows(), desc="Batch copying for all circuits"):
                    circuit = template_row['circuit']

                    # Filter valid rows for this circuit
                    valid_rows = original_data.loc[
                        (original_data['r_dateTimeHalfHour'].isin(valid_times)) &
                        (original_data['circuit'] == circuit)
                    ]

                    # Create a DataFrame that matches the template_row structure
                    mass_copied = pd.DataFrame([template_row.to_dict()] * len(gap_range))
                    mass_copied['r_dateTimeHalfHour'] = gap_range  # Update timestamps
                    mass_copied['nObs'] = 0  # Update nObs for gap rows

                    # Copy power columns from valid_rows into the mass_copied DataFrame
                    power_columns = ['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']
                    for column in power_columns:
                        mass_copied[column] = valid_rows[column].values  # Assign column values directly

                    # Append the mass_copied rows to the copied_rows list
                    copied_rows.extend(mass_copied.to_dict(orient='records'))

        if copied_rows:
            break
        
    if not copied_rows:  # If the loop completes without success
        print("Warning: Unable to fill gap with earlier years.")
    return copied_rows    

def copy_values_from_nearby_weeks_fallback(gap_range, matching_rows, original_data):
    
    """
    Fills gaps by copying values from nearby weeks for all circuits, using prevalidation with sampling and rigorous checking.
    Optimized to first identify valid times using one circuit and then perform batch copying for all circuits.
    """
    copied_rows = []

    # Reference circuit for checking data availability
    first_circuit = matching_rows.iloc[0]['circuit']
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
                (original_data['r_dateTimeHalfHour'] == candidate_time) &
                (original_data['circuit'] == first_circuit)
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
                    (original_data['r_dateTimeHalfHour'] == candidate_time) &
                    (original_data['circuit'] == first_circuit)
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
        circuit = template_row['circuit']

        # Filter valid rows for this circuit
        valid_rows_for_circuit = original_data.loc[
            (original_data['r_dateTimeHalfHour'].isin(valid_times)) &
            (original_data['circuit'] == circuit)
        ]

        # Create a DataFrame that matches the template_row structure
        mass_copied = pd.DataFrame([template_row.to_dict()] * len(gap_range))
        mass_copied['r_dateTimeHalfHour'] = gap_range  # Update timestamps
        mass_copied['nObs'] = 0  # Update nObs for gap rows

        # Copy power columns from valid_rows_for_circuit into the mass_copied DataFrame
        power_columns = ['meanPowerW', 'sdPowerW', 'minPowerW', 'maxPowerW']
        for column in power_columns:
            mass_copied[column] = valid_rows_for_circuit[column].values  # Assign column values directly

        # Append the mass_copied rows to the copied_rows list
        copied_rows.extend(mass_copied.to_dict(orient='records'))

    return copied_rows

# Function to fill NaN using nearby timestamps
def fill_missing_values_weatherdata(df):
    '''
    Used in the weatherdata generation function
    '''
    # Group by time of day and forward-fill, then backward-fill
    df['time_of_day'] = df['r_dateTimeHalfHour'].dt.time  # Extract the time of day
    df = df.sort_values('r_dateTimeHalfHour')  # Ensure data is sorted

    # Apply forward-fill and backward-fill grouped by time_of_day
    df['Radiation_MJ_per_m2'] = (
        df.groupby('time_of_day')['Radiation_MJ_per_m2']
        .apply(lambda x: x.ffill().bfill())
        .reset_index(level=0, drop=True)  # Reset index to align with the original DataFrame
    )
    df['Radiation_kW_per_m2'] = (
        df.groupby('time_of_day')['Radiation_kW_per_m2']
        .apply(lambda x: x.ffill().bfill())
        .reset_index(level=0, drop=True)  # Reset index to align with the original DataFrame
    )
    df['Windspeed_m_per_s'] = (
        df.groupby('time_of_day')['Windspeed_m_per_s']
        .apply(lambda x: x.ffill().bfill())
        .reset_index(level=0, drop=True)  # Reset index to align with the original DataFrame
    )
    df['Temperature_deg_C'] = (
        df.groupby('time_of_day')['Temperature_deg_C']
        .apply(lambda x: x.ffill().bfill())
        .reset_index(level=0, drop=True)  # Reset index to align with the original DataFrame
    )    
    # Drop helper column
    df.drop(columns=['time_of_day'], inplace=True)

    return df
