import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from pyomo.opt import SolverFactory
from tqdm import tqdm
from gap_filling_helpers import fill_missing_values_weatherdata
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pvlib

OUTPUT_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON_FOLDER_PATH = os.path.join(OUTPUT_FOLDER_PATH, "JSONs")
FOLDER_DATA_PATH = os.path.join(OUTPUT_FOLDER_PATH, r"Half hourly Hawkes bay\Processed_load_data_complete_dataset")

plt.rc('font', size=14)  # Default font size for text
plt.rc('axes', titlesize=16)  # Font size for titles
plt.rc('axes', labelsize=14)  # Font size for axis labels
plt.rc('legend', fontsize=14)  # Font size for legend


def Test_save_to_csv(output_file_path, output_dataframe):
    output_dataframe.to_csv(output_file_path, index=False)
    print(f"Processed data saved to {output_file_path}")
    return 
#* Preprocessing weather
def preprocess_weather_data(filepath, column_map, beginDT, endDT, columns_to_keep):
    """Load and preprocess weather data for radiation or wind."""
    beginDT = pd.to_datetime(beginDT, format='%d/%m/%Y').tz_localize('Pacific/Auckland')
    endDT = pd.to_datetime(endDT, format='%d/%m/%Y').tz_localize('Pacific/Auckland')
    
    data = pd.read_csv(filepath)
    data['OBS_DATE'] = pd.to_datetime(data['OBS_DATE'], utc=True)
    data['OBS_DATE'] = data['OBS_DATE'].dt.tz_convert('Pacific/Auckland')
    data.rename(columns=column_map, inplace=True)
    #data['r_dateTimeHalfHour'] = data['r_dateTimeHalfHour'].dt.tz_localize('Pacific/Auckland') #! None to 'Pacific/Auckland'??
    data.sort_values(by='r_dateTimeHalfHour', inplace=True)
    
    filtered_data = data[
        (data['r_dateTimeHalfHour'] >= beginDT) &
        (data['r_dateTimeHalfHour'] <= endDT)
    ] [columns_to_keep]
    
    return filtered_data

def resample_weather_data(filtered_data, resample_type):
    """Resample weather data based on the specified aggregation type."""
    filtered_data.set_index('r_dateTimeHalfHour', inplace=True)
    if resample_type == 'sum':
        resampled_data = filtered_data.resample('30min', label='right', closed='right').sum()
    elif resample_type == 'mean':
        resampled_data = filtered_data.resample('30min', label='right', closed='right').mean()
    resampled_data.drop(resampled_data.index[-1], axis=0, inplace=True)
    resampled_data.reset_index(inplace=True)
    return resampled_data

def Generate_weather_data(Gen_or_read, filepath_radiation, filepath_wind, filepath_temp, beginDT, endDT, DELETE_FEB_29, output_file_path):
    """
    Watch out for gaps in weather data!
    Create a dataframe with the relevant weather data, dropping all other information in the dataset. 
    Also interpolates the half hourly timestamps to match the load data timestamps.
    To later be used for generating power consumption data. 
    
    """ 
    TIME_INTERVAL_SECONDS = 30 * 60
    
    if Gen_or_read:
        beginDT = pd.to_datetime(beginDT, dayfirst=True)
        endDT = pd.to_datetime(endDT, dayfirst=True)

        # Preprocess radiation data
        column_map_rad = {'OBS_DATE': 'r_dateTimeHalfHour', 'RAD_GLOBAL10': 'Radiation_MJ_per_m2'}
        filtered_data_rad = preprocess_weather_data(filepath_radiation, column_map_rad, beginDT, endDT, ['r_dateTimeHalfHour', 'Radiation_MJ_per_m2'])
        df_rad = resample_weather_data(filtered_data_rad, 'sum')
        df_rad['Radiation_kW_per_m2'] = df_rad['Radiation_MJ_per_m2'] * 1000 / TIME_INTERVAL_SECONDS    # Preparing for detailed PV calculation
        print(f"length of df_Rad = {len(df_rad)}. Should be 17520 or 17568") 

        # Preprocess wind data
        column_map_wind = {'OBS_DATE': 'r_dateTimeHalfHour', 'MEAN_SPD10': 'Windspeed_m_per_s'}
        filtered_data_wind = preprocess_weather_data(filepath_wind, column_map_wind, beginDT, endDT, ['r_dateTimeHalfHour', 'Windspeed_m_per_s'])
        df_wind = resample_weather_data(filtered_data_wind, 'mean')
        print(f"length of df_wind = {len(df_wind)}. Should be 17520 or 17568")
        
        # Preprocess temperature data
        column_map_temp = {'OBS_DATE': 'r_dateTimeHalfHour', 'MEAN_TEMP10': 'Temperature_deg_C'}
        filtered_data_temp = preprocess_weather_data(filepath_temp, column_map_temp, beginDT, endDT, ['r_dateTimeHalfHour', 'Temperature_deg_C'])
        df_temp = resample_weather_data(filtered_data_temp, 'mean')
        print(f"length of df_temp = {len(df_temp)}. Should be 17520 or 17568")        

        # Merge radiation and wind and temp data
        Relevant_weather_data = pd.merge(df_rad, df_wind, on='r_dateTimeHalfHour', how='outer')
        Relevant_weather_data = pd.merge(Relevant_weather_data, df_temp, on='r_dateTimeHalfHour', how='outer')
        print(f"Length of the relevant_weather_data = {len(Relevant_weather_data)}")

        # Check and fill missing values
        if Relevant_weather_data.isnull().values.any():
            missing_rows = Relevant_weather_data[Relevant_weather_data.isnull().any(axis=1)]
            print("Found the following rows with missing values:")
            print(missing_rows)
            print("Proceeding to fill these using bfill and ffill, taking data from the same time...")
            Relevant_weather_data = fill_missing_values_weatherdata(Relevant_weather_data)
            if Relevant_weather_data.isnull().values.any():
                raise ValueError("Missing values detected in the relevant weather data.")    
        
        # Delete 29th feb
        if DELETE_FEB_29:
            Relevant_weather_data = Relevant_weather_data[~((Relevant_weather_data['r_dateTimeHalfHour'].dt.day == 29) & (Relevant_weather_data['r_dateTimeHalfHour'].dt.month == 2))].reset_index(drop=True)
            print(f"Length Relevant_weather_data after deleting 29th of February: {len(Relevant_weather_data)}")

        Test_save_to_csv(output_file_path, Relevant_weather_data)    
    else:
        Relevant_weather_data = pd.read_csv(output_file_path)
            
    return Relevant_weather_data

def Generate_pv_prod_profile_simple(Gen_or_read, Relevant_weather_data, pv_eff, pv_area_per_panel, output_file_path):
    """
    Production in W / panel installed
    Generates a seperate df_pv_prod based on the earlier selected relevant weather data. 
    For now this is done using a simple method. 
    --> Could add pv_nameplate_power to calculate per kW installed
    """
    if Gen_or_read:
        TIME_INTERVAL_SECONDS = 30 * 60
        df_pv_prod = Relevant_weather_data.copy()
        df_pv_prod['pv_production_kW'] = df_pv_prod['Radiation_MJ_per_m2'] * 1000 * pv_eff * pv_area_per_panel / TIME_INTERVAL_SECONDS #! be careful, working on alternative method and might have to adapt.
        df_pv_prod.drop(columns=['Radiation_MJ_per_m2', 'Windspeed_m_per_s'], inplace=True)
        Test_save_to_csv(output_file_path, df_pv_prod)
    else:
        df_pv_prod = pd.read_csv(output_file_path)
    
    return df_pv_prod

def Generate_pv_prod_profile_detailed(Relevant_weather_data, PV_area, PV_max_eff, PV_year1_eff, PV_Temp_coef, PV_tilt, PV_orientation):
    """
    Function to calculate PV production using PVlib.
    """
    STANDARD_TEMP = 25  # °C

    # Align indices by setting r_dateTimeHalfHour as the index
    #//Relevant_weather_data['r_dateTimeHalfHour'] = pd.to_datetime(Relevant_weather_data['r_dateTimeHalfHour']).dt.tz_localize('Pacific/Auckland', ambiguous = True, nonexistent = 'shift_forward')
    Relevant_weather_data.set_index('r_dateTimeHalfHour', inplace=True)
    
    # Extract relevant columns for calculation
    ghi = Relevant_weather_data['Radiation_kW_per_m2'] * 1000  # Convert kW/m² to W/m²
    temp_air = Relevant_weather_data['Temperature_deg_C']
    wind_speed = Relevant_weather_data['Windspeed_m_per_s']

    # Calculate solar position
    latitude = -37.94939  # Te pa o Penu as an example. Differences within Tairawhiti are negligible for the solar position. 
    longitude = 178.24875  
    altitude = 100  
    times = Relevant_weather_data.index

    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)

    # Decompose GHI into DNI and DHI using Erbs model
    dni_dhi = pvlib.irradiance.erbs(
        ghi=ghi,
        zenith=solpos['apparent_zenith'],
        datetime_or_doy=times
    )
    dni = dni_dhi['dni']
    dhi = dni_dhi['dhi']
    
    # Calculate POA (Plane of Array) irradiance
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PV_tilt,
        surface_azimuth=PV_orientation,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth']
    )

    poa_irradiance = poa['poa_global']

    # Calculate cell temperature
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['insulated_back_glass_polymer']
    # choose from: open_rack_glass_glass, close_mount_glass_glass, open_rack_glass_polymer, insulated_back_glass_polymer
    cell_temp = pvlib.temperature.sapm_cell(
        poa_irradiance, temp_air, wind_speed,
        **temperature_model_parameters
    )

    # Adjust efficiency based on temperature
    adjusted_efficiency = PV_max_eff/100 * PV_year1_eff/100 * (1 + PV_Temp_coef / 100 * (cell_temp - STANDARD_TEMP))

    # Calculate PV production in kW
    pv_production = poa_irradiance * adjusted_efficiency * PV_area / 1000  # Convert W to kW
    
    # Reset the index if r_dateTimeHalfHour is the index
    if 'r_dateTimeHalfHour' in Relevant_weather_data.index.names:
        Relevant_weather_data.reset_index(inplace=True)
        
    return pv_production

def Extend_production_profiles(df_prod, project_years, isPV, PV_degr):
    # Remove localization for extension
    df_prod['r_dateTimeHalfHour'] = pd.to_datetime(df_prod['r_dateTimeHalfHour'])
    df_prod['r_dateTimeHalfHour'] = df_prod['r_dateTimeHalfHour'].dt.tz_localize(None)
    
    # Calculate the start and end dates of the original data
    start_date = df_prod['r_dateTimeHalfHour'].min()
    end_date = df_prod['r_dateTimeHalfHour'].max()

    # Calculate the duration of the original data in years
    original_years = (end_date - start_date).days / 365
    original_years_raw = (end_date - start_date).days / 365
    original_years = round(original_years_raw) if abs(original_years_raw - round(original_years_raw)) < 0.02 else original_years_raw # force fix a strange bug that stems from timezones and leap years
    remaining_years = project_years - original_years
    num_full_copies = int(remaining_years // original_years)
    
    # Create the extended weather profile
    extended_data = []
    for i in range(num_full_copies + 2):
        # Shift the dates by the appropriate number of years
        shift_years = i * original_years
        shifted_data = df_prod.copy()
        shifted_data['r_dateTimeHalfHour'] = shifted_data['r_dateTimeHalfHour'] + pd.DateOffset(years=int(shift_years))
        extended_data.append(shifted_data)

    # Concatenate the extended data
    extended_data = pd.concat(extended_data, ignore_index=True)

    # Trim the data to match the desired length
    final_end_date = start_date + pd.DateOffset(years=project_years)
    extended_data = extended_data[extended_data['r_dateTimeHalfHour'] < final_end_date]
    
    if isPV:
        degraded_data = []
        for i in range(project_years+1):
            filtered_data = extended_data.copy()
            filtered_data = filtered_data[
                (filtered_data['r_dateTimeHalfHour'] >= (start_date + pd.DateOffset(years=int(i)))) &
                (filtered_data['r_dateTimeHalfHour'] < (start_date + pd.DateOffset(years=int(i+1))))
            ]
            filtered_data['pv_production_kW'] = filtered_data['pv_production_kW'] * ((1 - PV_degr / 100) ** i )
            degraded_data.append(filtered_data)
        # Concatenate the extended data
        extended_data = pd.concat(degraded_data, ignore_index=True)
            
    
    return extended_data
    # # pv
    # repeated_pvdata = np.tile(df_pv_prod.values, (project_years, 1))
    # df_pv_prod_extended = pd.DataFrame(repeated_pvdata, columns=df_pv_prod.columns)
    # df_pv_prod_extended.drop(columns='r_dateTimeHalfHour', inplace=True)    
    # # wind
    # repeated_winddata = np.tile(df_wind_prod.values, (project_years, 1))
    # df_wind_prod_extended = pd.DataFrame(repeated_winddata, columns=df_wind_prod.columns)
    # df_wind_prod_extended.drop(columns='r_dateTimeHalfHour', inplace=True)
    
    # print(f"Extended the timeseries data to the project lifetime. Removed the datetime column as the optimization model relies on index. ")
    # return df_pv_prod_extended, df_wind_prod_extended

def Generate_pv_prod_profile_complete(Gen_or_read, Relevant_weather_data, project_years, PV_area, PV_sys_eff, PV_max_eff, PV_year1_eff, PV_degr, PV_Temp_coef, PV_tilt, PV_tilt2, PV_orientation, PV_orientation2, PV_main_orient_perc, output_file_path):
    if Gen_or_read:
        if PV_main_orient_perc == 100:
            
            pv_production = Generate_pv_prod_profile_detailed(Relevant_weather_data, PV_area, PV_max_eff, PV_year1_eff, PV_Temp_coef, PV_tilt, PV_orientation)    
            #//print(f"first rows of pv_production {pv_production.head(48)}")    
        else:
            pv_production1 = Generate_pv_prod_profile_detailed(Relevant_weather_data, PV_area, PV_max_eff, PV_year1_eff, PV_Temp_coef, PV_tilt, PV_orientation)
            pv_production1 = pv_production1 * PV_main_orient_perc / 100
            
            pv_production2 = Generate_pv_prod_profile_detailed(Relevant_weather_data, PV_area, PV_max_eff, PV_year1_eff, PV_Temp_coef, PV_tilt2, PV_orientation2)
            pv_production2 = pv_production2 * (1 - PV_main_orient_perc/100) 
            
            pv_production = pv_production1 + pv_production2 
            #//print(f"first rows of pv_production {pv_production.head(48)}")
        
        # Add production to DataFrame
        df_pv_prod = Relevant_weather_data.copy()
        df_pv_prod['pv_production_kW'] = pv_production.values
        df_pv_prod['pv_production_kW'] = df_pv_prod['pv_production_kW'] * PV_sys_eff / 100
        
        # Reset the index if r_dateTimeHalfHour is the index
        if 'r_dateTimeHalfHour' in df_pv_prod.index.names:
            df_pv_prod.reset_index(inplace=True)
        
        df_pv_prod = Extend_production_profiles(df_pv_prod, project_years, True, PV_degr)
        
        df_pv_prod.drop(columns=['Radiation_MJ_per_m2', 'Radiation_kW_per_m2', 'Windspeed_m_per_s', 'Temperature_deg_C'], inplace=True)
        Test_save_to_csv(output_file_path, df_pv_prod)          
        
    else:
        df_pv_prod = pd.read_csv(output_file_path)
    return df_pv_prod

def Generate_wind_prod_profile_simple(Gen_or_read, Relevant_weather_data, project_years, path_power_curve, wind_gen_eff, wind_hub_height, output_file_path):
    """
    Production in W / installed turbine
    Generates a seperate df_wind_prod based on the earlier selected relevant weather data. 
    For now this is done using a simple method. 
    """
    if Gen_or_read:    
        ROUGNESS_LENGTH = 0.055
        
        df_wind_prod = Relevant_weather_data.copy()
        power_curve = pd.read_csv(path_power_curve)
        windspeeds_Powercurve_array = np.array(power_curve['Windspeed_m_per_s'])
        outputPower_Powercurve_array = np.array(power_curve['power_output_W'])

        df_wind_prod ['Windspeed_hub_m_per_s'] = df_wind_prod['Windspeed_m_per_s'] * np.log(wind_hub_height/ROUGNESS_LENGTH) / np.log(10/ROUGNESS_LENGTH)
        windspeeds_hub_array = np.array(df_wind_prod['Windspeed_hub_m_per_s'])

        poweroutput_turbine_array = np.interp(windspeeds_hub_array, windspeeds_Powercurve_array, outputPower_Powercurve_array) # Should linearly interpolate the values of windspeeds on the powercurve
        df_wind_prod ['wind_production_kW'] = poweroutput_turbine_array * wind_gen_eff * 0.001  # assuming that there is still a powerloss after the efficiency loss. 0.001 makes sure it is in kW.
        df_wind_prod.drop(columns=['Radiation_MJ_per_m2', 'Radiation_kW_per_m2', 'Temperature_deg_C', 'Windspeed_m_per_s', 'Windspeed_hub_m_per_s'], inplace=True)
        #extend profile
        df_wind_prod = Extend_production_profiles(df_wind_prod, project_years, False, 0)
        print(f"latest date in profile_wind_prod: {df_wind_prod['r_dateTimeHalfHour'].max()}")
        Test_save_to_csv(output_file_path, df_wind_prod)
    else:
        df_wind_prod = pd.read_csv(output_file_path)
    
    return df_wind_prod

#* Preprocessing load
def generate_annual_load_profiles_as_df(df_loadprofile, project_years, growth_rate, stochastic_factor):
    # Extract base profile
    base_profile = df_loadprofile['meanPowerkW'].values
    base_dates = pd.to_datetime(df_loadprofile['r_dateTimeHalfHour'])
    
    # Ensure base_dates are timezone-aware
    base_dates = base_dates.dt.tz_localize(None)        #! Removes localization == Naive timestamps.

    extended_profiles = [base_profile]
    extended_dates = [base_dates]
    
    for year in tqdm(range(1, project_years), desc="Extending the loadprofile according to project years"):
        # Generate the dates for this year
        year_dates = base_dates + pd.DateOffset(years=year)       
        # Adjust power profile
        stochastic_variation = np.random.normal(1, stochastic_factor/100, len(year_dates))
        growth_adjustment = (1 + growth_rate/100) ** year
        adjusted_profile = base_profile[:len(year_dates)] * stochastic_variation * growth_adjustment
        # Append to the extended lists
        extended_profiles.append(adjusted_profile)
        extended_dates.append(year_dates)
    
    # Flatten the lists and combine into a DataFrame
    final_dates = pd.concat(extended_dates).reset_index(drop=True)
    final_profiles = np.concatenate(extended_profiles)
    
    # Create the resulting DataFrame
    extended_df = pd.DataFrame({
        'r_dateTimeHalfHour': final_dates,
        'meanPowerkW': final_profiles
    })
    
    return extended_df

def simulate_marae_activity(df, scaling_factor, avg_weeks_btwn_act, seed=None):
    """
    Upscale the `meanPowerkW` for a random weekend (Friday, Saturday, Sunday) 
    on average every 4 weeks.

    Parameters:
    - df (pd.DataFrame): Input dataframe with `r_dateTimeHalfHour` and `meanPowerkW` columns.
    - scaling_factor (float): Factor by which to upscale the data.
    - avg_weeks_btwn_act (integer): Decides per how many weeks a weekend is upscaled.
    - start_date (str or None): Start date for the calculation. Defaults to the first date in the DataFrame.
    - seed (int or None): Random seed for reproducibility. Defaults to None.

    Returns:
    - pd.DataFrame: Dataframe with upscaled `meanPowerkW` for random weekends.
    """
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Ensure datetime format for r_dateTimeHalfHour
    df['r_dateTimeHalfHour'] = pd.to_datetime(df['r_dateTimeHalfHour'])
    df.sort_values(by='r_dateTimeHalfHour', inplace=True)
    
    # Calculate the 4-week block number for each timestamp (integer values)
    start_date = df['r_dateTimeHalfHour'].min()
    days_marae = 7*avg_weeks_btwn_act
    df['week_number'] = ((df['r_dateTimeHalfHour'] - pd.to_datetime(start_date)).dt.days // days_marae)

    # Group by each 4-week block
    grouped = df.groupby('week_number')

    # Randomly select 3 consecutive days from each 4-week block
    indices_to_upscale = []
    for _, group in grouped:
        # Extract unique days in the block
        unique_days = group['r_dateTimeHalfHour'].dt.date.drop_duplicates().values

        # Skip blocks with fewer than 3 days
        if len(unique_days) < 3:
            continue

        # Randomly select a starting day for the 3-day period
        random_start_idx = np.random.randint(0, len(unique_days) - 2)
        selected_days = unique_days[random_start_idx:random_start_idx + 3]

        # Collect the indices for the selected days
        for day in selected_days:
            indices_to_upscale.extend(group[group['r_dateTimeHalfHour'].dt.date == day].index)

    # Apply scaling factor to the selected rows
    df.loc[indices_to_upscale, 'meanPowerkW'] *= scaling_factor

    # Drop temporary columns
    df.drop(columns=['week_number'], inplace=True)

    return df

def Generate_cumulative_load_profile(Gen_or_read, loadprofile_components, project_years, growth_rate, stochastic_factor, marae_activity, marae_factor, marae_weeks, DELETE_FEB_29, output_file_path):
    """
    Generate a cumulative load profile compiled from selected households, circuits, and times. In kW
    """
    if Gen_or_read:
        # Create an empty cum_load_profile dataframe
        cum_load_profile = pd.DataFrame(columns=['r_dateTimeHalfHour', 'meanPowerW'])

        # Loop through each unique linkID
        for _, component in loadprofile_components.iterrows():
            linkID = component['linkID']
            alt_circuits = component['alt_circuits']
            circuit_mods = component['circuit_mods']
            N_Household = component['N_Household']
            beginDT = pd.to_datetime(component['beginDT'], format='%d/%m/%Y').tz_localize('Pacific/Auckland')
            endDT = pd.to_datetime(component['endDT'], format='%d/%m/%Y').tz_localize('Pacific/Auckland')
            Actual_kWh = component['Actual_yearly_kWh']

            # Construct file name 
            file_name = f"{linkID}_prepared_data.csv"
            file_path = os.path.join(FOLDER_DATA_PATH, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # load file and shift UTC to NZ time
            print(f"Processing file: {file_name}")
            data = pd.read_csv(file_path)
            data['r_dateTimeHalfHour'] = pd.to_datetime(data['r_dateTimeHalfHour'], utc=True)
            data['r_dateTimeHalfHour'] = data['r_dateTimeHalfHour'].dt.tz_convert('Pacific/Auckland')# Shifting the timezone 
            #data['r_dateTimeHalfHour'] = data['r_dateTimeHalfHour'].dt.tz_localize(None) #? Optional: making the datetimes timezone unaware again after shifting. 

            # Filter the data for the required time range and select the circuit column
            filtered_data = data[
                (data['r_dateTimeHalfHour'] >= beginDT) & 
                (data['r_dateTimeHalfHour'] < endDT)&       # Make sure that I do not get one extra
                (data['circuit'] == 'imputedTotalDemand_circuitsToSum_v1.1' )
            ][['r_dateTimeHalfHour', 'meanPowerW']]     # only requires the datetime and the meanPowerW of the data. 
            filtered_data.reset_index(drop=True, inplace=True)
            filtered_data = filtered_data.set_index('r_dateTimeHalfHour')   #? Debug: temporarily set index datetime to ensure proper modification
            
            # Making modifications to the load data
            for circuit, mod in zip(alt_circuits, circuit_mods):
                if circuit:
                    fdata = data[
                        (data['r_dateTimeHalfHour'] >= beginDT) & 
                        (data['r_dateTimeHalfHour'] < endDT)&       # Make sure that I do not get one extra
                        (data['circuit'] == circuit )
                    ][['r_dateTimeHalfHour', 'meanPowerW']]
                    fdata.reset_index(drop=True, inplace=True)                    
                    fdata = fdata.set_index('r_dateTimeHalfHour')           #? Debug: temporarily set index datetime to ensure proper modification
                    
                    fdata['meanPowerW'] = fdata['meanPowerW'] * mod
                    #filtered_data['meanPowerW'] += fdata['meanPowerW']
                    filtered_data['meanPowerW'] = filtered_data['meanPowerW'].add(fdata['meanPowerW'], fill_value=0) #? Debug
            filtered_data.reset_index(inplace=True)                         #? Debug: reset index
            
            # Multiply the circuit data by N_circuit and linearly scale to match actual consumption
            filtered_data['meanPowerkW'] = filtered_data['meanPowerW'] * 0.001 
            
            total_kWh_yearly = filtered_data['meanPowerkW'].sum() * 0.5
            print(f"total kWh of 1 modded {linkID} Household: {total_kWh_yearly}")
            if Actual_kWh:
                ScaleFactor = Actual_kWh / total_kWh_yearly
                filtered_data['meanPowerkW'] *= ScaleFactor
                total_kWh_yearly = filtered_data['meanPowerkW'].sum() * 0.5
                print(f"total kWh of 1 modded and scaled {linkID} Household: {total_kWh_yearly}")
            
            filtered_data['meanPowerkW'] *= N_Household

            # Ensure complete datetime coverage 
            if cum_load_profile.empty:
                cum_load_profile = filtered_data
            else:
                # Ensure timestamps match and throw informative error if needed #! this always gives an error
                if not cum_load_profile['r_dateTimeHalfHour'].equals(filtered_data['r_dateTimeHalfHour']):
                    # Find mismatches (only if error)
                    expected_times = set(cum_load_profile['r_dateTimeHalfHour'])
                    found_times = set(filtered_data['r_dateTimeHalfHour'])

                    # Identify missing and extra timestamps (only if error)
                    missing_in_filtered = expected_times - found_times
                    extra_in_filtered = found_times - expected_times

                    # Raise the error with detailed mismatch information
                    raise ValueError(
                        f"Mismatch in datetime coverage for {linkID}:\n"
                        f"Missing in filtered data: {list(missing_in_filtered)}...\n"
                        f"Extra in filtered data: {list(extra_in_filtered)}..."
                    )

                # Add the power values directly
                cum_load_profile['meanPowerkW'] += filtered_data['meanPowerkW']

        # Check the length of the cum loadprofile
        print(f"Expected 17520 entries, but got {len(cum_load_profile)}. ---> 2016 is a leap year, which adds 48 entries")
        print(f"Debugging -- checking type of r_dateTimeHalfHour: {cum_load_profile['r_dateTimeHalfHour'].dtype}")

        if DELETE_FEB_29:
            cum_load_profile = cum_load_profile[~((cum_load_profile['r_dateTimeHalfHour'].dt.day == 29) & (cum_load_profile['r_dateTimeHalfHour'].dt.month == 2))]
            print(f"Length loadprofile after deleting 29th of February: {len(cum_load_profile)}")

        # Ensure there are no missing values
        if cum_load_profile.isnull().values.any():
            print(f"Missing values:{cum_load_profile[cum_load_profile.isnull().any(axis=1)]}")
            raise ValueError("Missing values detected in the cumulative load profile.")

        # Generate 25 year long load profile and include marae activity if applicable
        ext_cum_load_profile = generate_annual_load_profiles_as_df(cum_load_profile, project_years, growth_rate, stochastic_factor)
        if marae_activity:
            ext_cum_load_profile = simulate_marae_activity(ext_cum_load_profile, marae_factor, marae_weeks)
        
        Test_save_to_csv(output_file_path, ext_cum_load_profile)
    else:
        ext_cum_load_profile = pd.read_csv(output_file_path)
    
    return ext_cum_load_profile    

#* Used to formulate the LCOE objective function
def calculate_annual_energy_consumed(df_extended, project_years, num_timesteps_per_year=17520):
    annual_energies = []
    for year in tqdm(range(project_years), desc="calculate_annual_energy_consumed"):
        start_idx = year * num_timesteps_per_year
        end_idx = start_idx + num_timesteps_per_year
        annual_energy = df_extended['meanPowerkW'].iloc[start_idx:end_idx].sum() * 0.5  # Convert half-hourly to kWh
        annual_energies.append(annual_energy)
    return annual_energies

def calculate_discount_factors(real_discount_rate, project_years):
    print("Running calculate_discount_factors")
    # for year in tqdm(range(project_years), desc="Calculating a list of discount factors"):
    #     Realdiscfactor = (1 + real_discount_rate) ** -year
    # print(f"Real discount factor: {Realdiscfactor}")
    return [(1 + real_discount_rate) ** -year for year in tqdm(range(project_years), desc="Calculating a list of discount factors")]

def calculate_total_discounted_energy(annual_energies, discount_factors):
    print("Running calculate_total_discounted_energy")
    total_disc_energy = sum(annual_energy * discount_factor for annual_energy, discount_factor in zip(annual_energies, discount_factors))
    print(f"total_discounted_energy: {total_disc_energy}")
    return total_disc_energy
#* Solver configuration
def configure_solver(solver_name, options):
    solver = SolverFactory(solver_name)
    for key, value in options.items():
        solver.options[key] = value
    return solver
#* JSONs
def makeJSON(variable, file_name):
    '''
    Converts arrays before being storing them in JSONs
    '''
    if isinstance(variable, np.ndarray):
        variable = variable.tolist()
    elif isinstance(variable, dict):
        # Recursively convert numpy arrays in dictionaries
        variable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in variable.items()}
    elif isinstance(variable, list):
        # Recursively convert numpy arrays in lists
        variable = [item.tolist() if isinstance(item, np.ndarray) else item for item in variable]

    file_path = os.path.join(OUTPUT_JSON_FOLDER_PATH,  file_name)
    if not os.path.isfile(file_path):
        os.makedirs(OUTPUT_JSON_FOLDER_PATH, exist_ok=True)
        print(f"'{file_name}' has been created in the folder '{OUTPUT_JSON_FOLDER_PATH}'.")
    with open(file_path, 'w') as f:
        json.dump(variable, f)
    print(f"'{file_name}' is saved in '{OUTPUT_JSON_FOLDER_PATH}'.")
    
def openJSON(file_name):
    '''
    Opens JSONs and stores them as python dictionaries or lists
    '''
    file_path = os.path.join(OUTPUT_JSON_FOLDER_PATH, file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"JSON not generated yet. Calculate {file_name} first.")
    try:
        with open(file_path, 'r') as f:
            loaded_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in file {file_name}: {e}")
    return loaded_data

#* results extraction - summary of solution
def extract_results(model, runtime):
    """
    Extracts optimization results into a dictionary for later processing.

    Parameters:
        model (ConcreteModel): The Pyomo model instance after optimization.
        runtime (float): The runtime of the optimization process (in seconds).

    Returns:
        dict: A dictionary containing key results such as component sizing,
              LCOE, total initial investment, and runtime.
    """
    results = {
        "pv_capacity": model.pv_capacity.value,
        "wind_capacity": model.wind_capacity.value,
        "battery_energy_capacity": model.battery_energy_capacity.value,
        "LCOE": model.LCOE.expr(),
        "initial_investment": model.initial_investment.expr(),
        "runtime": runtime
    }
    return results

def extract_time_series_results_modelMG(model, timesteps_to_save = None):
    """
    Extracts time series data for operational variables from the Pyomo model. The Renewable production can be constructed easily using Opt_results_dict and the original profile. same for load. 

    Parameters:
        model (ConcreteModel): The Pyomo model instance after optimization.

    Returns:
        dict: A dictionary containing lists of time series data for key operational variables.  
    """
    timesteps_to_process = range(0, timesteps_to_save) if timesteps_to_save else model.T
    
    time_series_data = {
        "P_battery_charge"      : [model.P_battery_charge[t].value for t in timesteps_to_process],
        "P_battery_discharge"   : [model.P_battery_discharge[t].value for t in timesteps_to_process],
        "battery_SoC_kWh"       : [model.battery_SoC_kWh[t].value for t in timesteps_to_process],
        "P_grid_import"         : [model.P_grid_import[t].value for t in timesteps_to_process],
        "P_grid_export"         : [model.P_grid_export[t].value for t in timesteps_to_process],
        "P_curtailed"           : [model.P_curtailed[t].value for t in timesteps_to_process]
    }
    return time_series_data

#* Making figures
#! testing:
def plot_combined_line_graph_from_json1(
    json_file_path,
    series_to_plot,
    output_file_path,
    ylabel,
    title,
    min_timesteps=None,
    max_timesteps=None,
    modifications=None,
    modifications2=None,
    secondary_y_series=None,
    secondary_y_series_same_axis = False,
    ylabel2 = None,
    csv_file_path=None,
    datalabel1 = None,
    datalabel2 = None
):
    """
    Create a combined line graph from a JSON file containing time series data.

    Args:
        json_file_path (str): Path to the JSON file.
        series_to_plot (list): List of keys to plot from the JSON.
        output_file_path (str, optional): Path to save the output graph image.
        min_timesteps (int, optional): Start index of the timesteps to plot.
        max_timesteps (int, optional): End index of the timesteps to plot.
        modifications (dict, optional): Mapping of series keys to modification functions.
        secondary_y_series (list, optional): List of series to plot on the secondary y-axis.
        

    Returns:
        None
    """
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Check if all keys are present in the JSON
    for key in series_to_plot:
        if key not in data:
            raise KeyError(f"Key '{key}' not found in the JSON file.")

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Apply modifications and plot primary y-axis data
    for key in series_to_plot:
        if datalabel1==None:
            datalabel1=key
        series_data = data[key]
        if modifications and key in modifications:
            series_data = [modifications[key](x) for x in series_data]
        print(f"indix range for this {key}: {min_timesteps}:{max_timesteps}")
        series_data = series_data[min_timesteps:max_timesteps]
        ax1.plot(series_data, label=datalabel1) #label=key

    if secondary_y_series_same_axis:
        data = pd.read_csv(csv_file_path)
        for key in secondary_y_series:
            series_data = data[key][min_timesteps:max_timesteps]
            if datalabel2==None:
                datalabel2=key
            if modifications2 and key in modifications2:
                series_data = [modifications2[key](x) for x in series_data]
            ax1.plot(series_data, linestyle="--", label=datalabel2)
        ax1.set_ylabel(datalabel2)
        
    ax1.set_title(title) 
    ax1.set_xlabel("Time Steps [Halfhours]")
    ax1.set_ylabel(ylabel) 
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot secondary y-axis data if specified
    if secondary_y_series and secondary_y_series_same_axis==False:
        ax2 = ax1.twinx()
        data = pd.read_csv(csv_file_path)
        for key in secondary_y_series:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in the csv file.")            
            series_data = data[key][min_timesteps:max_timesteps]
            datalabel2=key
            if modifications and key in modifications:
                series_data = [modifications[key](x) for x in series_data]
            ax2.plot(series_data, linestyle="--", label=f"{datalabel2} (Right side)")
        ax2.set_ylabel(ylabel2)
        ax2.legend(loc="upper right")

    # Save or display the plot
    if output_file_path:
        plt.savefig(output_file_path, dpi=300)
        print(f"Graph saved to {output_file_path}")
    plt.show()

def plot_combined_line_graph_from_json2(
    json_file_path,
    series_to_plot,
    output_file_path,
    ylabel,
    title,
    min_timesteps=None,
    max_timesteps=None,
    modifications=None,
    modifications2=None,
    secondary_y_series=None,
    secondary_y_series_same_axis=False,
    ylabel2=None,
    csv_file_path=None,
    datalabel1=None,
    datalabel2=None
):
    """Create a combined line graph from a JSON file containing time series data."""

    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Handle labels properly
    datalabel1_dict = {key: datalabel1 for key in series_to_plot} if isinstance(datalabel1, str) else (datalabel1 or {key: key for key in series_to_plot})

    # Plot primary y-axis data
    for key in series_to_plot:
        series_data = data[key]

        # Apply modifications if needed
        if modifications and key in modifications:
            series_data = [modifications[key](x) for x in series_data]

        # Slice data correctly
        sliced_series = series_data[min_timesteps:max_timesteps]
        time_steps = list(range(min_timesteps, max_timesteps))

        # Plot with correct X-axis values
        ax1.plot(time_steps, sliced_series, label=datalabel1_dict[key])

    # Plot secondary y-axis on the same axis
    if secondary_y_series_same_axis:
        data_df = pd.read_csv(csv_file_path)
        datalabel2_dict = {key: datalabel2 for key in secondary_y_series} if isinstance(datalabel2, str) else (datalabel2 or {key: key for key in secondary_y_series})

        for key in secondary_y_series:
            series_data = data_df[key].iloc[min_timesteps:max_timesteps]
            if modifications2 and key in modifications2:
                series_data = series_data.apply(modifications2[key])

            ax1.plot(time_steps, series_data, linestyle="--", label=datalabel2_dict[key])

        ax1.set_ylabel(ylabel2)

    ax1.set_title(title)
    ax1.set_xlabel("Time Steps [Halfhours]")
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot secondary y-axis separately if needed
    if secondary_y_series and not secondary_y_series_same_axis:
        ax2 = ax1.twinx()
        data_df = pd.read_csv(csv_file_path)
        datalabel2_dict = {key: datalabel2 for key in secondary_y_series} if isinstance(datalabel2, str) else (datalabel2 or {key: key for key in secondary_y_series})

        for key in secondary_y_series:
            series_data = data_df[key].iloc[min_timesteps:max_timesteps]
            if modifications and key in modifications:
                series_data = series_data.apply(modifications[key])

            ax2.plot(time_steps, series_data, linestyle="--", label=f"{datalabel2_dict[key]} (Right side)")

        ax2.set_ylabel(ylabel2)
        ax2.legend(loc="upper right")

    # Save or display the plot
    if output_file_path:
        plt.savefig(output_file_path, dpi=300)
    plt.show()

def plot_combined_line_graph_from_csv(
    csv_filepath,
    series_to_plot,
    output_file_path,
    ylabel,
    title,
    min_timesteps=None,
    max_timesteps=None,
    modifications=None,
    secondary_y_series=None,
    ylabel2=None
):
    """
    Create a combined line graph from a JSON file containing time series data.

    Args:
        json_file_path (str): Path to the JSON file.
        series_to_plot (list): List of keys to plot from the JSON.
        output_file_path (str, optional): Path to save the output graph image.
        min_timesteps (int, optional): Start index of the timesteps to plot.
        max_timesteps (int, optional): End index of the timesteps to plot.
        modifications (dict, optional): Mapping of series keys to modification functions.
        secondary_y_series (list, optional): List of series to plot on the secondary y-axis.
        

    Returns:
        None
    """
    df = pd.read_csv(csv_filepath)
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Apply modifications and plot primary y-axis data
    for key in series_to_plot:
        series_data = df[key]
        if modifications and key in modifications:
            series_data = [modifications[key](x) for x in series_data]
        series_data = series_data[min_timesteps:max_timesteps]
        ax1.plot(series_data, label=key)

    ax1.set_title(title) 
    ax1.set_xlabel("Time Steps [Halfhours]")
    ax1.set_ylabel(ylabel) 
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot secondary y-axis data if specified
    if secondary_y_series:
        ax2 = ax1.twinx()
        for key in secondary_y_series:           
            series_data = df[key][min_timesteps:max_timesteps]
            if modifications and key in modifications:
                series_data = [modifications[key](x) for x in series_data]
            ax2.plot(series_data, linestyle="--", label=f"{key} (Right side)")
        ax2.set_ylabel(ylabel2)
        ax2.legend(loc="upper right")

    # Save or display the plot
    if output_file_path:
        plt.savefig(output_file_path, dpi=300)
        print(f"Graph saved to {output_file_path}")
    plt.show()

    
#! Testing
def plot_load_profile(df, start_idx, end_idx, output_path, plot_title = "Load Profile for a full year"):
    """
    Plot the 'meanPowerkW' column of a DataFrame over a specified index range and save the plot.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'meanPowerkW' column.
        start_idx (int): The starting index for the plot.
        end_idx (int): The ending index for the plot.
        output_path (str): The full file path to save the plot.
    """
    if 'meanPowerkW' not in df.columns:
        raise ValueError("The DataFrame does not contain the column 'meanPowerkW'.")
    
    if start_idx < 0 or end_idx > len(df) or start_idx >= end_idx:
        raise ValueError("Invalid index range specified.")
    
    # Extract the relevant portion of the data
    data_to_plot = df['meanPowerkW'].iloc[start_idx:end_idx]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(data_to_plot, label='Mean Power (kW)')
    plt.title(plot_title)
    plt.xlabel("Index")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the output directory exists
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()
    
def plot_load_profiles_by_datetime(df, start_datetime, end_datetime, output_path, rf):
    """
    Plot the 'meanPowerkW' column of a DataFrame over a specified datetime range for multiple RFs and save the plots.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'meanPowerkW' and 'r_dateTimeHalfHour' columns.
        start_datetime (str): The starting datetime for the plot (format: "DD/MM/YYYY HH:MM").
        end_datetime (str): The ending datetime for the plot (format: "DD/MM/YYYY HH:MM").
        output_path (str): The directory to save the plots.
        rf_list (list): List of RF identifiers for dynamic file naming and titles.
    """
    if 'meanPowerkW' not in df.columns or 'r_dateTimeHalfHour' not in df.columns:
        raise ValueError("The DataFrame must contain 'meanPowerkW' and 'r_dateTimeHalfHour' columns.")
    
    # Ensure the datetime column is in datetime format
    df['r_dateTimeHalfHour'] = pd.to_datetime(df['r_dateTimeHalfHour'], format="%d/%m/%Y %H:%M")
    
    # Filter the data based on the datetime range
    filtered_df = df[(df['r_dateTimeHalfHour'] >= start_datetime) & (df['r_dateTimeHalfHour'] < end_datetime)]
    
    if filtered_df.empty:
        raise ValueError("No data available in the specified datetime range.")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Loop through the rf_list and create individual plots
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['r_dateTimeHalfHour'], filtered_df['meanPowerkW'], label='Mean Power (kW)')
    plt.title(f"Load profile for {rf}")
    plt.xlabel("Datetime")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)
    
    # Format X-axis ticks for readability
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))  # Show ticks every 2 hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))  # Format as 'HH:MM'
    plt.xticks(rotation=45, fontsize=10)  # Rotate and adjust font size
    
    # Set the background color explicitly to white
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # Save the plot with a dynamic filename
    output_file = os.path.join(output_path, f"load_profile_{rf}.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()  # Close the figure to save memory

def plot_pie_circuit_load_rf(linkID, beginDT, endDT, output_path): 
    beginDT = pd.to_datetime(beginDT, format='%d/%m/%Y').tz_localize('Pacific/Auckland')
    endDT = pd.to_datetime(endDT, format='%d/%m/%Y').tz_localize('Pacific/Auckland')
    
    # Construct file name 
    file_name = f"{linkID}_prepared_data.csv"
    file_path = os.path.join(FOLDER_DATA_PATH, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Processing file: {file_name}")
    data = pd.read_csv(file_path)
    data['r_dateTimeHalfHour'] = pd.to_datetime(data['r_dateTimeHalfHour'], utc=True)
    data['r_dateTimeHalfHour'] = data['r_dateTimeHalfHour'].dt.tz_convert('Pacific/Auckland')# Shifting the timezone 
    
    # Filter the data for the required time range and select the circuit column
    filtered_data = data[
        (data['r_dateTimeHalfHour'] >= beginDT) & 
        (data['r_dateTimeHalfHour'] < endDT)       # Make sure that I do not get one extra
    ][['r_dateTimeHalfHour', 'circuit', 'meanPowerW']]     # only requires the datetime and the meanPowerW of the data. 

    # Calculate total kWh for each circuit
    filtered_data['kWh'] = filtered_data['meanPowerW'] * 0.0005  # Convert W to kWh for half-hourly data
    circuit_kWh = filtered_data.groupby('circuit')['kWh'].sum()
    
    # Separate circuits starting with 'imputed' and 'incomer'
    imputed_total_kWh = circuit_kWh[circuit_kWh.index.str.startswith('imputed')].sum()
    circuit_kWh = circuit_kWh[~circuit_kWh.index.str.startswith(('imputed', 'Incomer'))]
    # Calculate the gap (remaining load not accounted for by specific circuits)
    total_other_kWh = imputed_total_kWh - circuit_kWh.sum()
        # Append the 'Other' category if there is a gap
    if total_other_kWh > 0:
        circuit_kWh['Other'] = total_other_kWh
    
    # filtering the labelling logic for small percentages
    def custom_autopct(pct):
        if pct < 3:  # If the percentage is below 3%, don't display the value
            return ""
        else:
            return f"{pct:.1f}%\n{(pct/100)*imputed_total_kWh:.1f} kWh"

        
    # Calculate percentages
    percentages = (circuit_kWh / imputed_total_kWh) * 100
    
    
    # Create labels with names
    labels = [f"{circuit}" for circuit in circuit_kWh.index]
    

    # Create labels with names, percentages, and absolute kWh values
    labels = [f"{circuit}\n{percent:.1f}%\n{energy:.1f} kWh"
              for circuit, percent, energy in zip(circuit_kWh.index, percentages, circuit_kWh)]
    
    # Plot the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        circuit_kWh,
        labels=[f"{circuit}" for circuit in circuit_kWh.index],
        startangle=90,
        autopct=custom_autopct,
        textprops={'fontsize': 9},
        labeldistance=1.05
    )
    plt.title(
    f"Annual load distribution of {linkID}\n"
    f"with a total annual consumption of {imputed_total_kWh:.1f} kWh",
    fontsize=14,
    fontweight='bold',
    loc='center'
)
    plt.ylabel("")  # Remove y-axis label for aesthetics
    plt.tight_layout()
    
    output_file = os.path.join(output_path, f"load_circuit_distribution_{linkID}.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()  # Close the figure to save memory