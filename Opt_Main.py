import os
# import tqdm
# import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import time
import Opt_helpers as Ohelp
import Opt_func as Ofunc
import gc
cwd = os.path.dirname(os.path.abspath(__file__))
paths = {
    "powercurve": os.path.join(cwd, r"Components and prices\Dartello_powercurve_5kw.csv"),
    "load"      : os.path.join(cwd, r"Profile_cum_load.csv"), 
    "radiation" : os.path.join(cwd, r"Weather data\24976_radiation_10min_Gis.csv"),
    "wind"      : os.path.join(cwd, r"Weather data\24976_wind_10min_Gis.csv"),
    "temp"      : os.path.join(cwd, r"Weather data\24976_screenobs_10min_Gis.csv"),
    "weather"   : os.path.join(cwd, r"Profile_weather_data.csv"),
    "pvprod"    : os.path.join(cwd, r"Profile_pv_prod.csv"),        
    "windprod"  : os.path.join(cwd, r"Profile_wind_prod.csv"),
    "MPSfile"   : os.path.join(cwd, r"model.mps"),
    "Solfiles"  : os.path.join(cwd, r"solution.sol"),
    "timeJson1" : os.path.join(cwd, r"JSONs\Opt_results_timeseries.json"),
    "graph1"    : os.path.join(cwd, r"Outputs_figures\graph1.jpg"),
    "graph2"    : os.path.join(cwd, r"Outputs_figures\graph2.jpg"),
    "graph3"    : os.path.join(cwd, r"Outputs_figures\graph3.jpg"),
    "graph4"    : os.path.join(cwd, r"Outputs_figures\graph3.jpg"),
    "fname_dict": "Opt_results_dict.json",                     
    "fname_time": "Opt_results_timeseries.json"                      
}

def sensitivity_runs(
    fname_dict="Opt_results_dict.json", 
    fname_time="Opt_results_timeseries.json", 
    project_years=25, 
    pv_cost=1100, 
    bat_cost=400,
    wind_cost=10000, 
    bat_rcost=330,
    bat_life=9,
    bat_RTE=83, 
    bat_soc_upper = 80,
    bat_soc_lower=20, 
    startDT_We="01/04/2015", 
    endDT_We="01/04/2023", 
    Nom_discount_rate = 5, 
    grid_conn=40,
    marae_scale=2,
    load_growth=1,
    pv_max=150,
    bat_max=150,
    genload=False
    ):
    # Get the script directory (cwd)
    
    #* Settings
    Main_ctrl = {   
    "GENERATE_LOAD_PROFILE"   : genload,
    "GENERATE_PV_WIND"        : True,
    "DELETE_FEB_29"           : True,
    "SIM_MARAE_ACTIVITY"      : True,
    "PROJECT_YEARS"           : project_years,                  
    }

    paths.update({
        "fname_dict": fname_dict,                     # old name "Opt_results_dict.json"
        "fname_time": fname_time                        # old name "Opt_results_timeseries.json"
    })

    Opt_config = {
        "Threads"       : 12,            #? Matching physical cores = better?
        "MIPGap"        : 0.01,         # Allow a 1% optimality gap for faster convergence. Should be okay because input inaccuracies are definitely higher than 1% as well.
        "Method"        : 2,            # -1 = Auto (Def); 0 = Primal simplex; 1= Dual simplex; 2 = Barrier - large continuous problems (not MIP) #? barrier only removes concurrent solving = faster. default (deterministic concurrent) leads to more robust answers
        "Heuristics"    : 0.2,          # 0.05 = Default, Min = 0 and Max = 1. Controls how much Solver spends on finding feasible solutions early in the solving process. More is faster but less accurate.
        "Timelimit"     : 9000,         #? Experimental: As first try was 30 minutes for a reasonable answer this should be good.
        'NodefileStart' : 0.5,          # inf = Default. Save memory by writing to disk after X GB
        'Presolve'      : -1,           # -1 = Auto (Def); 0 = no; 1 = Mild; 2 = Aggressive presolve
        'MIPFocus'      : 0,            # 0 = Balance (Def); 1 = Feasibility; 2 = Optimality; 3 = Bound improvement
        'Cuts'          : 3,            # Add more cutting planes to tighten the root node relaxation. -1 = Auto (Def); ... 3 = max
        'BarConvTol'    : 1e-4,         # 1e-8 (Def). Relax barrier convergence for speed
        'NumericFocus'  : 1,            # Handle numerical issues 
        'IISMethod'     : 1,            # Report why the constraints / bounds are infeasible
        # 'ResultFile'    : os.path.join(cwd, "model.mps"),  # Save as MPS file to save memory and time when running the same file with a different config
        # 'SolFiles'      : os.path.join(cwd, "solution.sol") # Save the solution to a .sol file as a backup and for further processing
    }  
    PVinfo = {
        "PV_max_eff"    : 22.2,         # [%]   Max efficiency as advertised on specsheets. PR-MAX3-420-BLK 22.2
        "PV_year1_eff"  : 98,           # [%]   Often referred to as "Year 1 minimum warranted output" on specsheets
        "PV_ann_degr"   : 0.25,         # [%]   Annual degredation after first year 
        "PV_sys_eff"    : 88.5,         # [%]   Systemlosses of the complete the PV system in whole percentages. 
        "PVarea"        : 1.046 * 1.812,# [m²]  Of Solarpower PR-MAX3-420-BLK  1.046 * 1.812
        "PVtempDeg"     : -0.27,        # [%/‘C]Power Temp Coeficient of PR-MAX3-420-BLK 0.27
        "PVorientPerc"  : 100,          # [%]   100% in case all panels will be oriented in the same way (e.g. 1 roof/field). In case of multiple roof orientations the percentage can be set (e.g. 50/50 east/west)
        "PVtilt"        : 40,           # [deg] Tilt of the panel relative to horizontal 
        "PVtilt2"       : 30,           # [deg] Tilt of the second panel orientation. No effect if PVorientPerc == 100
        "PVorient"      : 0,            # [deg] Orientation (azimuth) of the solar panels. 0 = (True) North, 90 = East etc. 
        "PVorient2"     : 90,           # [deg] Orientation (azimuth) of the second panel orientation, e.g. in case of east/west setup, or when using 2 roofs. No effect if PVorientPerc == 100     
    }

    CompInfo = {
        "WINDeff"       : 0.95,         # [%]   Placeholder generator efficiency
        "HubHeight"     : 10,           # [m]   Height of wind turbine
        "Grid_conn_size": grid_conn,    # [kW]  Maximum size of the transformer. kW = typical powerfactor (0.8 worst case) * kVA
        "Bat_RTE"       : bat_RTE,      # [%]   Assumed constant over project lenght. Taken into account in assumption. 
        "Bat_SoC_upper" : bat_soc_upper,# [%]   Upper boundary of battery SoC to maximize longevity. 
        "Bat_SoC_lower" : bat_soc_lower,# [%]   Lower boundary of battery SoC to maximize longevity.
        "Bat_Lifespan"  : bat_life,     # [yrs] Expected year for replacement of battery cells. 9 years means that it has to be replaced on the first of the 10th year
        "Bat_max_Crate" : 0.5,          # [kW/kWh] Fixed Power to Energy ratio for the batteries, sets the maximum battery power capacity based on the optimized (thus variable) energy capacity. 
        "Inv_Lifespan"  : 15,           # [yrs] Expected year for replacement of Hybrid inverter
        "startDT_We"    : startDT_We,   #       Start date of selected weather data 
        "endDT_We"      : endDT_We,     #       End date of selected weather data
        "Load_growth"   : load_growth,  # [%]   Annual load growth rate
        "Load_stoch"    : 3,            # [%]   Every year there are slight variations in load per halfhour. Equals standard deviation.
        "MaraeWeeks"    : 4,            # [int] Average amount of weeks inbetween marae activity. Has to be an integer. Activity is randomly chosen within these Maraeweeks and lasts 3 consecutive days
        "MaraeScaleFact": marae_scale   # [ ]   Assumed that the aggregate community load will increase with this scalar. Likely between 1-1.5, as it is assumed that even though a lot of people come, all communnity member are also present and will thus not have significant electricity use. 
    }

    CostParam = {
        "Nom_discount_rate"         : Nom_discount_rate,# %    
        "pv_cost_per_panel"         : pv_cost,          # $/panel installed. Includes HYBRID inverter, controller, optimizers and mounting. 
        "wind_cost_per_turbine"     : wind_cost,        # $/turbine installed. Including controller.    
        "battery_ins_cost_per_kwh"  : bat_cost,         # $/kWh installed capacity. Includes ancillary equipment and C&C (labour). Inverter is already considered in PV_cost per panel.   
        "battery_cell_repl_per_kwh" : bat_rcost,        # $/kWh replaced batterycell capacity. Ancillary equipment not considered. Labour included. 
        "inverter_repl_per_panel"   : 280,              # $/panel installed. Replaces HYBRID inverter. Labour included. 
        "energy_import_price"       : 0.404,            # $/kWh from grid electricity tarrif, set at a constant price for now
        "import_price_trendline_a"  : 0.8115,           #       Y=a*x+b. Linear regression over the past 21 years. 
        "energy_export_price"       : 0.08,             # $/kWh to grid buyback rate/tarrif. Set to 0 for a "battery first" strategy that doesn't trade on the electricity market for profits
        "export_price_growth_rate"  : 0.00              # %/year starting from the second year
    }
 

    start_runtime = time.time()                         # Record the start time for runtime measurement
    #* Define load profile components
    #! rf_41 only has data from 28/05/2016 01:00 
    #! If applicable, modify the loadprofile of that household by entering a list [string] of alt_circuits and its corresponding modifications in the same order for circuit_mods [float]
    #! If known, fill in 'Actual_yearly_kWh. The halfhourly loads will be linearly scaled AFTER the circuits have been substracted / adjusted.
    load_profile_info = pd.DataFrame(columns=['beginDT', 'endDT', 'linkID', 'alt_circuits', 'circuit_mods', 'N_Household', 'Actual_yearly_kWh'])
    load_profile_info.loc[0] = ['01/04/2015', '01/04/2016', 'rf_34', [f'Heat Pump$4223'] , [-1],  1, 6100]  
    load_profile_info.loc[1] = ['01/04/2015', '01/04/2016', 'rf_37', [f'Heat Pump$4134',f'Laundry & Fridge Freezer$4138'] , [-1, 2],  1, None]
    load_profile_info.loc[2] = ['01/04/2015', '01/04/2016', 'rf_34', [f'Heat Pump$4223',f'Hot Water - Uncontrolled$4224', f'Laundry & Garage Freezer$4227'] , [-1, 1, 1],  1, None]
    load_profile_info.loc[3] = ['01/04/2015', '01/04/2016', 'rf_42', [f'Heat Pump$4130',f'Hot Water - Uncontrolled$4131', f'Laundry & Freezer$4128', f'Lighting (inc heat lamps)$4129'] , [-1, 0.2, 1, -1],  1, 14000]
    load_profile_info.loc[4] = ['01/04/2015', '01/04/2016', 'rf_34', [f'imputedTotalDemand_circuitsToSum_v1.1', f'Lighting$4222'] , [-1,1],  1, None]  
    load_profile_info.loc[5] = ['01/04/2015', '01/04/2016', 'rf_44', [f'Heat Pump$4154'] , [-1],  1, None]
    load_profile_info.loc[6] = ['01/04/2015', '01/04/2016', 'rf_32', [f'Heat Pump$4196', f'Hot Water - Controlled$4198', f'Kitchen Appliances$4195'] , [-1, 0.5, 0.25],  1, None]

    #* Generate / read the timeseries data   
    df_load_profile = Ohelp.Generate_cumulative_load_profile(Main_ctrl["GENERATE_LOAD_PROFILE"], load_profile_info, Main_ctrl["PROJECT_YEARS"], CompInfo["Load_growth"], CompInfo["Load_stoch"], Main_ctrl["SIM_MARAE_ACTIVITY"], CompInfo["MaraeScaleFact"], CompInfo["MaraeWeeks"], Main_ctrl["DELETE_FEB_29"], paths["load"])
    # Ohelp.plot_load_profile(df_load_profile, 0, 17520, paths["graph3"], plot_title="Load profile for a full year of the RTHH") #13200, 30720
    # break
    weather_data = Ohelp.Generate_weather_data(Main_ctrl["GENERATE_PV_WIND"], paths["radiation"], paths["wind"], paths["temp"], CompInfo["startDT_We"], CompInfo["endDT_We"], Main_ctrl["DELETE_FEB_29"], paths["weather"])
    df_pv_prod = Ohelp.Generate_pv_prod_profile_complete(Main_ctrl["GENERATE_PV_WIND"], weather_data, Main_ctrl["PROJECT_YEARS"], PVinfo["PVarea"], PVinfo["PV_sys_eff"], PVinfo["PV_max_eff"], PVinfo["PV_year1_eff"], PVinfo["PV_ann_degr"], PVinfo["PVtempDeg"], PVinfo["PVtilt"], PVinfo["PVtilt2"], PVinfo["PVorient"], PVinfo["PVorient2"], PVinfo["PVorientPerc"], paths["pvprod"])
    df_wind_prod = Ohelp.Generate_wind_prod_profile_simple(Main_ctrl["GENERATE_PV_WIND"], weather_data, Main_ctrl["PROJECT_YEARS"], paths["powercurve"], CompInfo["WINDeff"], CompInfo["HubHeight"], paths["windprod"])

    #* Run the optimization model
    print("Now setting up the optimizer....")
    opt = Ohelp.configure_solver('gurobi', Opt_config) 
    Model_MG = Ofunc.create_microgrid_model(
        df_loadprofile  = df_load_profile,
        df_pv_prod      = df_pv_prod,
        df_wind_prod    = df_wind_prod,
        CostParams      = CostParam,
        CompInfo        = CompInfo,
        pv_max_capacity = pv_max,
        battery_max_capacity = bat_max,    
        project_years   = Main_ctrl["PROJECT_YEARS"]    
    )

    print(f"Optimizer set up. Solving...")
    opt.solve(Model_MG, tee=True)

    # Extract timeseries and store as a dict of lists
    Opt_results_timeseries = Ohelp.extract_time_series_results_modelMG(Model_MG)   
    Ohelp.makeJSON(Opt_results_timeseries, paths["fname_time"]) 

    # Print the total runtime 
    total_runtime = time.time() - start_runtime
    print(f"Script runtime: {total_runtime:.2f} seconds")

    # Extract results and store in JSON dict
    Opt_results_dict = Ohelp.extract_results(Model_MG, total_runtime)  
    Ohelp.makeJSON(Opt_results_dict, paths["fname_dict"])
    del Model_MG
    gc.collect()
    return df_load_profile, paths

print("starting runs")

# #* Run 1 baseline
# sensitivity_runs(fname_dict="run1p_dict_default.json", fname_time="run1p_time_default.json")
# print("run 1 baseline finished")

# #* Runs: Discountfactors
# sensitivity_runs(fname_dict="run2p_dict_1disc.json", fname_time="run2p_time_1disc.json", Nom_discount_rate = 1)
# print("run 2p 1disc finished")
# sensitivity_runs(fname_dict="run2p_dict_2disc.json", fname_time="run2p_time_2disc.json", Nom_discount_rate = 2)
# print("run 2p 2disc finished")
# sensitivity_runs(fname_dict="run2p_dict_3disc.json", fname_time="run2p_time_3disc.json", Nom_discount_rate = 3)
# print("run 2p 3disc finished")
# sensitivity_runs(fname_dict="run2p_dict_4disc.json", fname_time="run2p_time_4dsic.json", Nom_discount_rate = 4)
# print("run 2p 4disc finished")
# sensitivity_runs(fname_dict="run2p_dict_6disc.json", fname_time="run2p_time_6dsic.json", Nom_discount_rate = 6)
# print("run 2p 6disc finished")
# sensitivity_runs(fname_dict="run2p_dict_7disc.json", fname_time="run2p_time_7dsic.json", Nom_discount_rate = 7)
# print("run 2p 7disc finished")
# sensitivity_runs(fname_dict="run2p_dict_8disc.json", fname_time="run2p_time_8dsic.json", Nom_discount_rate = 8)
# print("run 2p 8disc finished")

# #* Runs: pricing
# sensitivity_runs(fname_dict="run3p_dict_1cost.json", fname_time="run3p_time_1cost.json", pv_cost = 700, bat_cost = 300)
# print("run 3p 1cost finished")
# sensitivity_runs(fname_dict="run3p_dict_2cost.json", fname_time="run3p_time_2cost.json", pv_cost = 800, bat_cost = 325)
# print("run 3p 2cost finished")
# sensitivity_runs(fname_dict="run3p_dict_3cost.json", fname_time="run3p_time_3cost.json", pv_cost = 900, bat_cost = 350)
# print("run 3p 3cost finished")
# sensitivity_runs(fname_dict="run3p_dict_4cost.json", fname_time="run3p_time_4cost.json", pv_cost = 1000, bat_cost = 375)
# print("run 3p 4cost finished")
# sensitivity_runs(fname_dict="run3p_dict_5cost.json", fname_time="run3p_time_5cost.json", pv_cost = 1200, bat_cost = 425)
# print("run 3p 5cost finished")
# sensitivity_runs(fname_dict="run3p_dict_6cost.json", fname_time="run3p_time_6cost.json", pv_cost = 1300, bat_cost = 450)
# print("run 3p 6cost finished")

# #* Runs: dry and wet
# sensitivity_runs(fname_dict="run4p_dict_1sun.json", fname_time="run4p_time_1sun.json", startDT_We="01/04/2015", endDT_We="01/04/2016")
# print("run 4p 1sun finished")
# sensitivity_runs(fname_dict="run4p_dict_2wet.json", fname_time="run4p_time_2wet.json", startDT_We="01/04/2018", endDT_We="01/04/2019")
# print("run 4p 2wet finished")
# sensitivity_runs(fname_dict="run4p_dict_3cloudy.json", fname_time="run4p_time_3cloudy.json", startDT_We="01/04/2022", endDT_We="01/04/2023")
# print("run 4p 3cloudy finished")

# #* Runs: project lifetime
# sensitivity_runs(fname_dict="run5p_dict_10years.json", fname_time="run5p_time_10years.json", project_years = 10, genload=True)
# print("run 5p 10y finished")
# sensitivity_runs(fname_dict="run5p_dict_15years.json", fname_time="run5p_time_15years.json", project_years = 15, genload=True)
# print("run 5p 15y finished")
# sensitivity_runs(fname_dict="run5p_dict_20years.json", fname_time="run5p_time_20years.json", project_years = 20, genload=True)
# print("run 5p 20y finished")
# sensitivity_runs(fname_dict="run5p_dict_30years.json", fname_time="run5p_time_30years.json", project_years = 30, genload=True)
# print("run 5p 30y finished")

#* Runs: Other load adjustments
# sensitivity_runs(fname_dict="run6p_dict_marae1.json", fname_time="run6p_dict_marae1.json", marae_scale=1, genload=True )
# print("run 6p marae1 finished")
# sensitivity_runs(fname_dict="run6p_dict_marae1_5.json", fname_time="run6p_dict_marae1_5.json", marae_scale=1.5, genload=True )
# print("run 6p marae1_5 finished")
# sensitivity_runs(fname_dict="run6p_dict_marae3.json", fname_time="run6p_dict_marae3.json", marae_scale=3, genload=True )
# print("run 6p marae3 finished")
# sensitivity_runs(fname_dict="run6p_dict_Lgrowth0.json", fname_time="run6p_dict_Lgrowth0.json", load_growth=0, genload=True )
# print("run 6p Lgrowth0 finished")

#* Runs: Miscellaneous
# sensitivity_runs(fname_dict="run7p_dict_1newbat.json", fname_time="run7p_time_1newbat.json", bat_soc_upper=100, bat_cost=860, bat_rcost=730, bat_life=20, bat_RTE=93)
# print("run 7p 1newbat finished")
# sensitivity_runs(fname_dict="run7p_dict_2pvmax.json", fname_time="run7p_time_2pvmax.json", pv_max=50)
# print("run 7p 2pvmax finished")
# sensitivity_runs(fname_dict="run7p_dict_3pvmax.json", fname_time="run7p_time_3pvmax.json", pv_max=100) 
# print("run 7p 3pvmax finished")
# sensitivity_runs(fname_dict="run7p_dict_4offgrid.json", fname_time="run7p_time_4offgrid.json", pv_max=300, bat_max= 300, grid_conn=0) 
# print("run 7p 4offgrid finished")
# sensitivity_runs(fname_dict="run7p_dict_5maxgrid.json", fname_time="run7p_time_5maxgrid.json", grid_conn=100) 
# print("run 7p 5maxgrid finished")


#* Runs: Reducing winds costs until it becomes useful
# sensitivity_runs(fname_dict="run8p_dict_1wind.json", fname_time="run8p_time_1wind.json", wind_cost=8000)
# print("run 8p 1wind finished")
# sensitivity_runs(fname_dict="run8p_dict_2wind.json", fname_time="run8p_time_2wind.json", wind_cost=7000)
# print("run 8p 2wind finished")
# sensitivity_runs(fname_dict="run8p_dict_3wind.json", fname_time="run8p_time_3wind.json", wind_cost=6000)
# print("run 8p 3wind finished")
# sensitivity_runs(fname_dict="run8p_dict_4wind.json", fname_time="run8p_time_4wind.json", wind_cost=5000)
# print("run 8p 4wind finished")
# sensitivity_runs(fname_dict="run8p_dict_5wind.json", fname_time="run8p_time_5wind.json", wind_cost=4000)
# print("run 8p 5wind finished")
# sensitivity_runs(fname_dict="run8p_dict_6wind.json", fname_time="run8p_time_6wind.json", wind_cost=3000)
# print("run 8p 6wind finished")
# sensitivity_runs(fname_dict="run8p_dict_7wind.json", fname_time="run8p_time_7wind.json", wind_cost=2000)
# print("run 8p 7wind finished")

# * Runs: model validation
sensitivity_runs(genload=False, project_years=1, pv_max= 18, bat_max=20.4, 
                 bat_RTE=93, bat_soc_upper=100, bat_soc_lower=37, startDT_We="01/01/2024", endDT_We="01/01/2025")
print("run run9V 1 finished")
# #* Plotting (Temp)
# #todo Put the filename as a variable in these functions
# #todo make these functions better / more generalized
# Ohelp.plot_load_profile(df_load_profile, 13200, 30720, paths["graph3"]) #13200, 30720
# #* Making plots, temporary location
# series_to_plot_1 = ["P_grid_import", "P_grid_export"]
# series_to_plot_2 = ["battery_SoC_kWh"]
# series_to_plot_3 = ["P_curtailed"]
# modifications={"battery_SoC_kWh": lambda x: x / 0.204}         # Normalize SoC to 100% Mintimestep: 17520 - (2160*2 = 4320) = 13200 (1st jan) Maxtimestep: 13200 + 17520 = 30720 [1] and 13200 + (744*2) = 14688 #// 
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_1, paths["graph1"] ,"Grid import and production surplus (kW)" , "Microgrid behaviour of 1 year, starting in january", min_timesteps=13200, max_timesteps=30720)
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, paths["graph2"] , "Battery State of Charge (%)", "Battery behaviour in January", 13200, 14688)
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_3, paths["graph4"] ,"Curtailment (kW)" , "Microgrid curtailment of 1 year, starting in january", min_timesteps=13200, max_timesteps=30720)
