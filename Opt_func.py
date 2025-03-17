from pyomo.environ import *
import pandas as pd
import numpy as np
import Opt_helpers as Ohelp

def create_microgrid_model(
    df_loadprofile,
    df_pv_prod,
    df_wind_prod,
    CostParams,
    CompInfo,
    num_timesteps=17520,    
    pv_max_capacity=150,   
    wind_max_capacity=30,   
    battery_max_capacity=150,      
    grid_connection_size=100,        
    project_years=25               
):  
    # Create model
    print("Starting model creation...")
    model = ConcreteModel()
    
    #*SETS
    model.T = RangeSet(0, num_timesteps * project_years -1, doc="Timesteps for the complete project (default is 25*17520 - 1) (halfhourly)")
    model.Y = RangeSet(0, project_years - 1, doc="Years in the project")
    battery_replacement_years = [y for y in range(project_years) if y % CompInfo["Bat_Lifespan"] == 0 and y > 0]
    inverter_replacement_years = [y for y in range(project_years) if y % CompInfo["Inv_Lifespan"] == 0 and y > 0]    
    
    #* PARAMETERS      
    model.time_series_load = Param(model.T, initialize=dict(enumerate(df_loadprofile['meanPowerkW'])))
    model.time_series_pv = Param(model.T, initialize=dict(enumerate(df_pv_prod['pv_production_kW'])))
    model.time_series_wind = Param(model.T, initialize=dict(enumerate(df_wind_prod['wind_production_kW'])))
    
        # Financial parameters
    NOM_DISCOUNT_RATE = CostParams["Nom_discount_rate"]/100
    DISCOUNT_FACTORS = Ohelp.calculate_discount_factors(NOM_DISCOUNT_RATE, project_years)
    
        # Costs (NZD$) == PLACEHOLDERS
    model.pv_cost_per_panel = Param(initialize=CostParams["pv_cost_per_panel"], doc="$/panel installed, includes inverters, controller, optimizers and mounting")              
    model.wind_cost_per_turbine = Param(initialize=CostParams["wind_cost_per_turbine"], doc="$/turbine installed, including controller")    
    model.battery_ins_cost_per_kwh = Param(initialize=CostParams["battery_ins_cost_per_kwh"], doc="$/kWh installed energy capacity")
    
    model.grid_energy_costs = Param(model.Y, initialize={year: CostParams["energy_import_price"] + (CostParams["import_price_trendline_a"]/100) * year
        for year in range(project_years)}, doc="$/kWh from grid, adjusted per year")
    model.buyback_rates = Param(model.Y, initialize={year: CostParams["energy_export_price"] * (1 + CostParams["export_price_growth_rate"]) ** year
        for year in range(project_years)}, doc="$/kWh to grid, adjusted per year")
    
        # Replacement/repair costs and timing   
    model.battery_cell_repl_per_kwh = Param(initialize=CostParams["battery_cell_repl_per_kwh"], doc="$/kWh replaced energy capacity")
    model.inverter_repl_per_panel = Param(initialize=CostParams["inverter_repl_per_panel"], doc="$/panel replaced inverter capacity")
    
        # Technical parameters == PLACEHOLDERS
    model.battery_efficiency = Param(initialize=(CompInfo["Bat_RTE"]/100), doc="Round-trip efficiency. Now constant, could degrade as well")    
    model.battery_power_ratio = Param(initialize=CompInfo["Bat_max_Crate"], doc="constant kW/kWh ratio of the battery to keep the battery characteristics realistic")  
    grid_connection_size = CompInfo["Grid_conn_size"]
    model.grid_connection_size = Param(initialize=CompInfo["Grid_conn_size"], doc="Maximum grid connection size in kW, limits the power import and export")
    
            
    #* DECISION VARIABLES
        # Component sizing
    model.pv_capacity = Var(
        bounds=(pv_max_capacity, pv_max_capacity),              #! bounds 1, pv_max and initialize 50
        initialize=pv_max_capacity,      # trying to speed up the process
        domain = NonNegativeReals,          #! Changed from NonNegativeIntegers to make it non-MIP
        doc='# PV panels installed capacity (0.420 kW per panel)'
    ) 
    
    model.wind_capacity = Var(
        bounds=(0, wind_max_capacity),
        initialize=1,       # trying to speed up the process
        domain = NonNegativeReals,       #! Changed from NonNegativeIntegers to make it non-MIP
        doc='# Wind turbines installed (5.5 kW per turbine)' 
    )
    
    model.battery_energy_capacity = Var(
        bounds=(battery_max_capacity, battery_max_capacity),
        initialize=battery_max_capacity,     # trying to speed up the process
        doc='Battery energy capacity (kWh)'
    )
    
        # Operational variables - defined for each timestep and useful to plot what is happening
    model.P_battery_charge = Var(
        model.T,
        bounds=(0, None), 
        initialize=0,
        doc='Battery charging power (kW)'
    )
    
    model.P_battery_discharge = Var(          
        model.T,
        bounds=(0, None),
        initialize=0,
        doc='Battery discharging power (kW)'
    )
    
    model.battery_SoC_kWh = Var(               
        model.T,
        bounds=(0, None), 
        initialize=0,
        doc='Battery energy level (kWh)'     
    )
    
    model.P_grid_import = Var(
        model.T,
        bounds=(0, grid_connection_size), 
        initialize=0,
        doc='Power imported from grid (kW)'
    )
    
    model.P_grid_export = Var(            
        model.T,
        bounds=(0, grid_connection_size),
        initialize=0,    
        doc="Production surplus (kW) at each timestep when generation exceeds load and storage is full. Could be an income stream if export_price_growth_rate > 0."
    )
    
    model.P_curtailed = Var(
        model.T,
        bounds=(0, None),
        initialize=0,    
        doc="Production surplus (kW) at each timestep when generation exceeds load, storage is full and there is no more room to export electricity" 
    )
        
    #* CONSTRAINTS
        # Power balance constraint for each timestep
    def power_balance_rule(model, t):                                                                   
        return (model.pv_capacity * model.time_series_pv[t] + model.wind_capacity * model.time_series_wind[t] +
                model.P_grid_import[t] + model.P_battery_discharge[t] == model.P_battery_charge[t] + model.time_series_load[t] + model.P_grid_export[t] + model.P_curtailed[t])
    model.power_balance = Constraint(
        model.T,
        rule=power_balance_rule,
        doc="Ensure power balance is kept at all times"
    )
    
    def curtailment_activation_rule(model, t):
        return model.P_curtailed[t] <= model.grid_connection_size - model.P_grid_export[t]
    model.curtailment_rule = Constraint(
        model.T,
        rule=curtailment_activation_rule,
        doc="Ensure curtailment is only activated when P_grid_export has reached its limits. Relevent if buyback rate = 0"
    )
    
    #* Battery constraints
        # Split SoC capacity constraint into two constraints
        # Lower bound: Ensure SoC is above 20% of energy capacity
    def battery_soc_capacity_lower_rule(model, t):
        return model.battery_SoC_kWh[t] >= (CompInfo["Bat_SoC_lower"]/100) * model.battery_energy_capacity  
    model.battery_soc_capacity_lower = Constraint(
        model.T, 
        rule=battery_soc_capacity_lower_rule,
        doc="Ensure SoC remains above 20% of energy capacity"
    )

        # Upper bound: Ensure SoC is below 80% of energy capacity
    def battery_soc_capacity_upper_rule(model, t):
        return model.battery_SoC_kWh[t] <= (CompInfo["Bat_SoC_upper"]/100) * model.battery_energy_capacity  
    model.battery_soc_capacity_upper = Constraint(
        model.T, 
        rule=battery_soc_capacity_upper_rule,
        doc="Ensure SoC remains below 80% of energy capacity"
    )

        # Avoid simultaneous charge and discharge
    model.charge_indicator = Var(model.T, domain=Binary, doc="1 if charging, 0 if discharging")
    
    def avoid_simultaneous_charge_rule(model, t):
        return (model.P_battery_charge[t] <= model.battery_power_ratio * model.battery_energy_capacity * model.charge_indicator[t])            
    model.avoid_simultaneous_charge = Constraint(model.T, rule=avoid_simultaneous_charge_rule)              

    def avoid_simultaneous_discharge_rule(model, t):
        return (model.P_battery_discharge[t] <= model.battery_power_ratio * model.battery_energy_capacity * (1 - model.charge_indicator[t]))   
    model.avoid_simultaneous_discharge = Constraint(model.T, rule=avoid_simultaneous_discharge_rule)


        # Updating SoC
    def soc_update_rule_kWh (model, t):
        if t == 0:
            return Constraint.Skip  # Initialisation of battery_SoC_kWh happens in the variable. 
        return (
            model.battery_SoC_kWh[t] ==
            model.battery_SoC_kWh[t-1] +
            0.5 * model.P_battery_charge[t] * model.battery_efficiency - # Halfhourly timesteps, thus 0.5*kW=kWh. ! And full round trip efficiency accounted for during charging (Doesn't make a difference as the value of the energy becomes relevant as soon as it is discharged, so the energylevel would now show how much energy could be deployed)
            0.5 * model.P_battery_discharge[t]
        )
    model.soc_update_kWh = Constraint(
        model.T,
        rule=soc_update_rule_kWh,
        doc="Update battery energy level with losses during charging only")
            

    #* Objective Function
        # Investment Costs as Expressions
    model.pv_investment_cost = Expression(
        expr=model.pv_cost_per_panel * model.pv_capacity,
        doc="Total investment cost for PV panels"
    )
    model.wind_investment_cost = Expression(
        expr=model.wind_cost_per_turbine * model.wind_capacity,
        doc="Total investment cost for wind turbines"
    )
    model.battery_investment_cost = Expression(         
        expr=model.battery_ins_cost_per_kwh * model.battery_energy_capacity,
        doc="Total investment cost for batteries"
    )

        # Total Initial Investment as Expression 
    model.initial_investment = Expression(
        expr=model.pv_investment_cost + model.wind_investment_cost + model.battery_investment_cost,
        doc="Total initial investment for the microgrid"
    )

        # Annual Energy Produced = annual energy consumed (neglecting P_grid_export). Calculated outside of the model.     
    annual_energies_consumed = Ohelp.calculate_annual_energy_consumed(df_loadprofile, project_years) # Generates a list of energy consumed per year
    total_discounted_energy = Ohelp.calculate_total_discounted_energy(annual_energies_consumed, DISCOUNT_FACTORS) # Generates 1 number of total discounted energy
            
    def total_discounted_costs_rule(model):
        battery_replacement_costs = sum(
            model.battery_cell_repl_per_kwh * model.battery_energy_capacity * DISCOUNT_FACTORS[year]
            for year in battery_replacement_years
        ) if battery_replacement_years else 0

        inverter_replacement_costs = sum(
            model.inverter_repl_per_panel * model.pv_capacity * DISCOUNT_FACTORS[year]
            for year in inverter_replacement_years
        ) if inverter_replacement_years else 0

        operational_costs = sum(
            sum(
                (
                    model.grid_energy_costs[year] * model.P_grid_import[t] -
                    model.buyback_rates[year] * model.P_grid_export[t]
                ) * 0.5 * DISCOUNT_FACTORS[year]
                for t in range(year * 17520, (year + 1) * 17520)
            )
            for year in model.Y
        )

        return model.initial_investment + operational_costs + battery_replacement_costs + inverter_replacement_costs

    model.total_discounted_costs = Expression(
        rule=total_discounted_costs_rule, 
        doc="Total discounted costs over the project lifetime"
    )

        #* Objective Function: Minimize LCOE
    def lcoe_rule(model):
        return model.total_discounted_costs / total_discounted_energy
    model.LCOE = Objective(rule=lcoe_rule, sense=minimize, doc="Levelized cost of electricity")
    
    if not isinstance(model, ConcreteModel):  # Ensure a valid Pyomo model is returned
        raise ValueError("Failed to create the microgrid model.")
    
    return model
