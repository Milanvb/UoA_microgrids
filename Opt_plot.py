import Opt_helpers as Ohelp
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
cwd = os.path.dirname(os.path.abspath(__file__)) 

paths = {
    "dailyprofile"      : os.path.join(cwd, r"Outputs_figures\Daily_profile_per_rf\current_daily_profile.csv"),
    "loadprofile"       : os.path.join(cwd, r"Validation data 1 household\2024validation_load_profile.csv"),
    "validation"        : os.path.join(cwd, r"Validation data 1 household\2024validation_prepared_data.csv"),
    "timeJson1"         : os.path.join(cwd, r"JSONs\V3 sensitivity runs\run1p_time_default.json"),
    "timeJson2"         : os.path.join(cwd, r"JSONs\Run9v_time_2.json"),
    "daily_hh_graphs"   : os.path.join(cwd, r"Outputs_figures\Daily_profile_per_rf"),
    "graph1"            : os.path.join(cwd, r"Outputs_figures\graph1.jpg"),
    "graph2"            : os.path.join(cwd, r"Outputs_figures\graph2.jpg"),
    "graph3"            : os.path.join(cwd, r"Outputs_figures\graph3.jpg"), 
    "graph4"            : os.path.join(cwd, r"Outputs_figures\graph4.jpg"), 
}

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

series_to_plot_1 = ["P_grid_import", "P_grid_export"]
series_to_plot_2 = ["battery_SoC_kWh"]
series_to_plot_3 = ["P_grid_import", "P_grid_export", "P_battery_charge", "P_battery_discharge"]
modifications={"battery_SoC_kWh": lambda x: x / 0.874}
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_3,paths["graph1"] ,"Grid import and export (kW)" , "Microgrid behaviour of 1 week in january", min_timesteps=13296, max_timesteps=13344)
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, paths["graph2"] , "Battery State of Charge (%)", "Battery behaviour in January", 13200, 14688)
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_1,paths["graph3"] ,"Grid import and export (kW)" , "Microgrid behaviour in january", min_timesteps=13200, max_timesteps=14688)
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_3,paths["graph4"] ,"Grid import and export (kW)" , "Microgrid behaviour of 1 week in August", min_timesteps=23376, max_timesteps=27412)
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, paths["graph4"] , "Battery State of Charge (%)", "Battery behaviour in August", 23376, 24863)
#Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_3,paths["graph4"] ,"Curtailment (kW)" , "Microgrid curtailment of 1 year, starting in january", min_timesteps=13200, max_timesteps=30720)


exit()






# Data from the table
data = {
    "Discount": [1, 2, 3 ,4 , 5, 6 ,7],
    "LCOE": [0.41, 0.4147, 0.4123, 0.4228, 0.45, 0.4164, 0.416],
    "PV": [90.3, 94.9, 89.5, 112.9, 50, 100, 99],
    "Battery": [67.8, 71.3, 67.2, 78.2, 87.4, 87.4, 73.3],
    "Investment": [21075, 22150, 20879, 31919, 14994, 24161, 23061]
}

df = pd.DataFrame(data)

# Baseline values (DEFAULT case)
baseline = {
    "LCOE": 0.4162,
    "PV": 106.3,
    "Battery": 87.4,
    "Investment": 25315
}

# Normalize data by dividing by the baseline
df_norm = df.copy()
df_norm["LCOE"] = df["LCOE"] / baseline["LCOE"]
df_norm["PV"] = df["PV"] / baseline["PV"]
df_norm["Battery"] = df["Battery"] / baseline["Battery"]
df_norm["Investment"] = (df["Investment"] / baseline["Investment"])  # Still normalized

# Convert investment to thousands for easier readability
df["Investment"] = df["Investment"] / 1000
baseline["Investment"] = baseline["Investment"] / 1000

# Define pastel colors
colors = ["#aec7e8", "#98df8a", "#ff9896", "#c5b0d5"]  # Light blue, green, red, purple

# Adding the baseline scenario (normalized to 1)
df_norm.loc[-1] = [0] + [1, 1, 1, 1]  # Baseline values (normalized)
df_norm.sort_index(inplace=True)

# Create a grouped bar chart with normalized values but absolute value labels
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(df_norm["Discount"]))

# Plot bars for each metric
bars_lcoe = ax.bar(x - 1.5 * bar_width, df_norm["LCOE"], bar_width, label="LCOE [$]", color=colors[0])
bars_pv = ax.bar(x - 0.5 * bar_width, df_norm["PV"], bar_width, label="PV Size [# panels]", color=colors[1])
bars_battery = ax.bar(x + 0.5 * bar_width, df_norm["Battery"], bar_width, label="Battery Size [kWh]", color=colors[2])
bars_investment = ax.bar(x + 1.5 * bar_width, df_norm["Investment"], bar_width, label="Invested/hh [x$1000]", color=colors[3])

# Labels and legend
ax.set_xlabel("Scenario code")
ax.set_ylabel("Normalized Values (Relative to Baseline)")
ax.set_xticks(x)
ax.set_xticklabels(["Baseline"] + df["Discount"].astype(str).tolist()) #
ax.set_title("Sensitivity Analysis: various scenarios", fontweight="bold")
ax.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Extend y-axis to ensure data labels fit
ax.set_ylim(0, max(df_norm.max()) * 0.31)

# Adding absolute value labels above bars
for bars, metric in zip([bars_lcoe, bars_pv, bars_battery, bars_investment], df.columns[1:]):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        abs_value = df[metric][i - 1] if i > 0 else baseline[metric]  # Use baseline for first label
        
        # Apply 2 significant figures ONLY to LCOE, rest are rounded integers
        if metric == "LCOE":
            formatted_value = f"{abs_value:.2g}"
        else:
            formatted_value = f"{int(round(abs_value))}"

        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, 
                formatted_value, ha="center", va="bottom", fontsize=10, 
                fontweight="bold", rotation=45, color="black")

# Show the plot
plt.show()



exit()
#* Generate a file that contains a small version of the load profile
# Define load profile components

#rf_list = ['rf_29', 'rf_30', 'rf_31', 'rf_32', 'rf_33', 'rf_34', 'rf_36', 'rf_37', 'rf_39', 'rf_42', 'rf_44', 'rf_45', 'rf_46', 'rf_47']
# rf_list = ['rf_47']
# GEN_OR_READ = True
# DELETE_FEB29 = True

# beginDT    = '01/04/2015'
# endDT       = '01/04/2016'

# for rf in tqdm(rf_list, desc="Cycling through all households"):
#     Ohelp.plot_pie_circuit_load_rf(rf, beginDT, endDT, paths['daily_hh_graphs'])
Timeseries = Ohelp.openJSON(paths["timeJson1"])

P_grid_import_annual = Timeseries["P_grid_import"]
P_grid_import_annual_sum = sum(P_grid_import_annual)/2
print(P_grid_import_annual_sum)

P_grid_export_annual = Timeseries["P_grid_export"]
P_grid_export_annual_sum = sum(P_grid_export_annual)/2
print(P_grid_export_annual_sum)

P_battery_charge_annual = Timeseries["P_battery_charge"]
P_battery_charge_annual_sum = sum(P_battery_charge_annual)/2
print(P_battery_charge_annual_sum)

P_battery_discharge_annual = Timeseries["P_battery_discharge"]
P_battery_discharge_annual_sum = sum(P_battery_discharge_annual)/2
print(P_battery_discharge_annual_sum)

#plots
series_to_plot_1 = ["P_grid_import", "P_grid_export"] 
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_model_imex_y.jpg") ,"Grid import and export (kW)" , 
                                          "Model outputs for 1 year", min_timesteps=0, max_timesteps=17520)

modifications={"P_grid_export": lambda x: x * -1} 
Ohelp.plot_combined_line_graph_from_csv(paths["validation"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_real_imex_y.jpg") ,"Grid import and export (kW)" , "RTHH data for 1 year", min_timesteps=0, max_timesteps=17520, modifications=modifications)


#* jan
start = 0
end = 1487 #exclusive

P_grid_import_jan = Timeseries["P_grid_import"]
P_grid_import_jan_sum = sum(P_grid_import_jan[start:end+1])/2
print(f"P_grid_import_jan_sum {P_grid_import_jan_sum}")

P_grid_export_jan = Timeseries["P_grid_export"]
P_grid_export_jan_sum = sum(P_grid_export_jan[start:end+1])/2
print(f"P_grid_export_jan_sum {P_grid_export_jan_sum}")

P_battery_charge_jan = Timeseries["P_battery_charge"]
P_battery_charge_jan_sum = sum(P_battery_charge_jan[start:end+1])/2
print(f"P_grid_export_jan_sum {P_battery_charge_jan_sum}")

P_battery_discharge_jan = Timeseries["P_battery_discharge"]
P_battery_discharge_jan_sum = sum(P_battery_discharge_jan[start:end+1])/2
print(f"P_battery_discharge_jan_sum {P_battery_discharge_jan_sum}")

#plots
modifications={"battery_SoC_kWh": lambda x: x / 0.204} 
series_to_plot_2 = ["battery_SoC_kWh"]
secondary_y_series = ["battery_SoC_perc"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, os.path.join(cwd, r"Outputs_figures\v_bat_jan.jpg") , "Modelled battery State of Charge (%)", "Battery behaviour in the 1st week of January", min_timesteps=0, max_timesteps=336, 
                                          modifications=modifications, secondary_y_series=secondary_y_series, ylabel2="RTHH battery State of Charge (%)", csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled battery", datalabel2="RTHH battery")

# series_to_plot_1 = ["P_grid_import", "P_grid_export"] 
# modifications2={"P_grid_export": lambda x: x * -1} 
series_to_plot_1 = ["P_grid_import"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_imex_jan.jpg") ,"Grid import" , "Model outputs for 1st week of January", min_timesteps=0, max_timesteps=336, 
                                          secondary_y_series=series_to_plot_1, ylabel2="RTHH data for 1st week of January", csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled import", datalabel2="RTHH import")
# exit()
# modifications={"P_grid_export": lambda x: x * -1} 
# Ohelp.plot_combined_line_graph_from_csv(paths["validation"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_real_imex_jan.jpg") ,"Grid import and export (kW)" , "RTHH data for 1st week of January", min_timesteps=0, max_timesteps=335, 
#                                         modifications=modifications)


#* Aug
start = 10178
end = 11665 #exclusive
P_grid_import_aug = Timeseries["P_grid_import"]
P_grid_import_aug_sum = sum(P_grid_import_aug[start:end+1])/2
print(f"P_grid_import_aug_sum {P_grid_import_aug_sum}")

P_grid_export_aug = Timeseries["P_grid_export"]
P_grid_export_aug_sum = sum(P_grid_export_aug[start:end+1])/2
print(f"P_grid_export_aug_sum {P_grid_export_aug_sum}")

P_battery_charge_aug = Timeseries["P_battery_charge"]
P_battery_charge_aug_sum = sum(P_battery_charge_aug[start:end+1])/2
print(f"P_grid_export_aug_sum {P_battery_charge_aug_sum}")

P_battery_discharge_aug = Timeseries["P_battery_discharge"]
P_battery_discharge_aug_sum = sum(P_battery_discharge_aug[start:end+1])/2
print(f"P_battery_discharge_aug_sum {P_battery_discharge_aug_sum}")

# modifications={"battery_SoC_kWh": lambda x: x / 0.204} 
# series_to_plot_2 = ["battery_SoC_kWh"]
# secondary_y_series = ["battery_SoC_perc"]
# Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, os.path.join(cwd, r"Outputs_figures\v_bat_aug.jpg") , "Modelled battery State of Charge (%)", "Battery behaviour in the 1st week of August", min_timesteps=10370, max_timesteps=10706, modifications=modifications, secondary_y_series=secondary_y_series, ylabel2="RTHH battery State of Charge (%)", csv_file_path=paths["validation"])

# modifications={"P_grid_export": lambda x: x * -1} 
# Ohelp.plot_combined_line_graph_from_csv(paths["validation"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_real_imex_aug.jpg") ,"Grid import and export (kW)" , "RTHH data for 1st week of August", min_timesteps=10370, max_timesteps=10706, modifications=modifications)

modifications={"battery_SoC_kWh": lambda x: x / 0.204} 
series_to_plot_2 = ["battery_SoC_kWh"]
secondary_y_series = ["battery_SoC_perc"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_2, os.path.join(cwd, r"Outputs_figures\v_bat_aug.jpg") , "Modelled battery State of Charge (%)", "Battery behaviour in the 1st week of August", min_timesteps=10370, max_timesteps=10706, 
                                          modifications=modifications, secondary_y_series=secondary_y_series, ylabel2="RTHH battery State of Charge (%)", csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled battery", datalabel2="RTHH battery")

# series_to_plot_1 = ["P_grid_import", "P_grid_export"] 
# modifications2={"P_grid_export": lambda x: x * -1} 
series_to_plot_1 = ["P_grid_import"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson1"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_model_im_aug.jpg") ,"Grid import (kW)" , "Model outputs for 1st week of August", min_timesteps=10370, max_timesteps=10706, 
                                          secondary_y_series=series_to_plot_1, csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled import", datalabel2="RTHH import")

#* REAL PV AS INPUT
Timeseries = Ohelp.openJSON(paths["timeJson2"])

P_grid_import_annual = Timeseries["P_grid_import"]
P_grid_import_annual_sum = sum(P_grid_import_annual)/2
print(P_grid_import_annual_sum)

P_grid_export_annual = Timeseries["P_grid_export"]
P_grid_export_annual_sum = sum(P_grid_export_annual)/2
print(P_grid_export_annual_sum)

P_battery_charge_annual = Timeseries["P_battery_charge"]
P_battery_charge_annual_sum = sum(P_battery_charge_annual)/2
print(P_battery_charge_annual_sum)

P_battery_discharge_annual = Timeseries["P_battery_discharge"]
P_battery_discharge_annual_sum = sum(P_battery_discharge_annual)/2
print(P_battery_discharge_annual_sum)

#* jan
start = 0
end = 1487 #exclusive

P_grid_import_jan = Timeseries["P_grid_import"]
P_grid_import_jan_sum = sum(P_grid_import_jan[start:end+1])/2
print(f"P_grid_import_jan_sum {P_grid_import_jan_sum}")

P_grid_export_jan = Timeseries["P_grid_export"]
P_grid_export_jan_sum = sum(P_grid_export_jan[start:end+1])/2
print(f"P_grid_export_jan_sum {P_grid_export_jan_sum}")

P_battery_charge_jan = Timeseries["P_battery_charge"]
P_battery_charge_jan_sum = sum(P_battery_charge_jan[start:end+1])/2
print(f"P_grid_export_jan_sum {P_battery_charge_jan_sum}")

P_battery_discharge_jan = Timeseries["P_battery_discharge"]
P_battery_discharge_jan_sum = sum(P_battery_discharge_jan[start:end+1])/2
print(f"P_battery_discharge_jan_sum {P_battery_discharge_jan_sum}")

modifications={"battery_SoC_kWh": lambda x: x / 0.204} 
series_to_plot_2 = ["battery_SoC_kWh"]
secondary_y_series = ["battery_SoC_perc"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson2"], series_to_plot_2, os.path.join(cwd, r"Outputs_figures\v_bat_jan_PV.jpg") , "Modelled battery State of Charge (%)", "Battery behaviour in the 1st week of January with real PV", min_timesteps=0, max_timesteps=335, 
                                          modifications=modifications, secondary_y_series=secondary_y_series, ylabel2="RTHH battery State of Charge (%)", csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled battery", datalabel2="RTHH battery")

# series_to_plot_1 = ["P_grid_import", "P_grid_export"] 
# modifications2={"P_grid_export": lambda x: x * -1} 
series_to_plot_1 = ["P_grid_import"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson2"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_imex_jan_PV.jpg") ,"Grid import and export (kW)" , "Model outputs for 1st week of January with real PV", min_timesteps=0, max_timesteps=335, 
                                          secondary_y_series=series_to_plot_1, ylabel2="RTHH data for 1st week of January", csv_file_path=paths["validation"], secondary_y_series_same_axis=True)


#* Aug
start = 10178
end = 11665 #exclusive
P_grid_import_aug = Timeseries["P_grid_import"]
P_grid_import_aug_sum = sum(P_grid_import_aug[start:end+1])/2
print(f"P_grid_import_aug_sum {P_grid_import_aug_sum}")

P_grid_export_aug = Timeseries["P_grid_export"]
P_grid_export_aug_sum = sum(P_grid_export_aug[start:end+1])/2
print(f"P_grid_export_aug_sum {P_grid_export_aug_sum}")

P_battery_charge_aug = Timeseries["P_battery_charge"]
P_battery_charge_aug_sum = sum(P_battery_charge_aug[start:end+1])/2
print(f"P_grid_export_aug_sum {P_battery_charge_aug_sum}")

P_battery_discharge_aug = Timeseries["P_battery_discharge"]
P_battery_discharge_aug_sum = sum(P_battery_discharge_aug[start:end+1])/2
print(f"P_battery_discharge_aug_sum {P_battery_discharge_aug_sum}")

modifications={"battery_SoC_kWh": lambda x: x / 0.204} 
series_to_plot_2 = ["battery_SoC_kWh"]
secondary_y_series = ["battery_SoC_perc"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson2"], series_to_plot_2, os.path.join(cwd, r"Outputs_figures\v_bat_aug_PV.jpg") , "Modelled battery State of Charge (%)", "Battery behaviour in the 1st week of August", min_timesteps=10370, max_timesteps=10706, 
                                          modifications=modifications, secondary_y_series=secondary_y_series, ylabel2="RTHH battery State of Charge (%)", csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled battery", datalabel2="RTHH battery")

# series_to_plot_1 = ["P_grid_import", "P_grid_export"] 
# modifications2={"P_grid_export": lambda x: x * -1} 
series_to_plot_1 = ["P_grid_import"]
Ohelp.plot_combined_line_graph_from_json2(paths["timeJson2"], series_to_plot_1, os.path.join(cwd, r"Outputs_figures\v_im_aug_PV.jpg") ,"Grid import (kW)" , "Model outputs for 1st week of August", min_timesteps=10370, max_timesteps=10706, 
                                          secondary_y_series=series_to_plot_1, csv_file_path=paths["validation"], secondary_y_series_same_axis=True, datalabel1="Modelled import", datalabel2="RTHH import")
