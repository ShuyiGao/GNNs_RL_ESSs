"""
Pyomo-based model for the 34-node active
distribution network (ADN) benchmark.
"""

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    SolverFactory, minimize,
)


def construct_opf_model(Vnom, Vmin, Vmax, Data_Network): # 1.0 0.3 0.3
    # Data Processing
    battery_parameters = {
        'capacity': 0.5, # MW.h
        'max_charge': 0.2, # MW
        'max_discharge': 0.2, # MW
        'efficiency': 1,
        'degradation': 0,  # euro/kw
        'max_soc': 0.8,
        'min_soc': 0.2,
        'initial_soc': 0.4}
    TIMES = Data_Network['TIMES']
    NODES = Data_Network['NODES']
    LINES = Data_Network['LINES']
    Tb = Data_Network['Tb']
    PD = Data_Network['PD']
    QD = Data_Network['QD']
    R = Data_Network['R']
    X = Data_Network['X']
    BATTERY_NODES = Data_Network['BATTERY_NODES']
    PRICE = Data_Network['PRICE']

    # Type of Model
    model = ConcreteModel()

    # Define Sets
    model.NODES = Set(initialize=NODES)
    model.LINES = Set(initialize=LINES)
    model.TIMES = Set(initialize=TIMES)
    # Define Parameters
    model.Vnom = Param(initialize=Vnom, mutable=False)
    model.Vmin = Param(initialize=Vmin, mutable=False)
    model.Vmax = Param(initialize=Vmax, mutable=False)
    # Tb_fixed = {key + 1: value for key, value in Tb.items()}
    model.Tb = Param(model.NODES, initialize=Tb, mutable=True)
    model.QD = Param(model.TIMES,model.NODES, initialize=0, mutable=True)  # Node demand
    model.R = Param(model.LINES, initialize=R, mutable=False)  # Line resistance
    model.X = Param(model.LINES, initialize=X, mutable=False)  # Line resistance
    # Define parameters for battery
    model.battery_initial_soc = Param(default=battery_parameters['initial_soc'])
    model.battery_capacity = Param(default=battery_parameters['capacity'])
    model.battery_soc_max = Param(default=battery_parameters['max_soc'])
    model.battery_soc_min = Param(default=battery_parameters['min_soc'])
    model.battery_max_change = Param(default=battery_parameters['max_charge'])


    # Define initialize PD
    def PD_init_rule(model,time,node):
        model.PD[time,node]=PD[time,node]
        return (model.PD[time,node])
    model.PD=Param(model.TIMES,model.NODES,initialize=PD_init_rule)

    def R_init_rule(model, i, j):
        return (model.R[i, j])

    model.RM = Param(model.LINES, initialize=R_init_rule)  # Line resistance

    def X_init_rule(model, i, j):
        return (model.X[i, j])

    model.XM = Param(model.LINES, initialize=X_init_rule)  # Line resistance

    # Define Variables
    model.P = Var(model.TIMES,model.LINES, initialize=0)  # Acive power flowing in lines
    model.Q = Var(model.TIMES,model.LINES, initialize=0)  # Reacive power flowing in lines
    model.I = Var(model.TIMES,model.LINES, initialize=0)  # Current of lines

    model.SOC=Var(model.TIMES,model.NODES,initialize=model.battery_initial_soc,bounds=(model.battery_soc_min,model.battery_soc_max))

    def energy_change_rule(model, time, i):
        if i not in BATTERY_NODES:
            tem = 0.0
            model.energy_change[time, i].fixed = True
        else:
            tem = 0.0
            model.energy_change[time, i].fixed = False
        return tem

    model.energy_change=Var(model.TIMES,model.NODES,initialize=energy_change_rule,bounds=(-model.battery_max_change,model.battery_max_change))

    def PS_init_rule(model, time,i):
        if model.Tb[i].value == 0:
            temp = 0.0
            model.PS[time,i].fixed = True
        else:
            temp = 0.0
        return temp
    model.PS = Var(model.TIMES,model.NODES, initialize=PS_init_rule)  # Active power of the SS

    def QS_init_rule(model,time,i):
        if model.Tb[i].value == 0:
            temp = 0.0
            model.QS[time,i].fixed = True
        else:
            temp = 0.0
        return temp
    model.QS = Var(model.TIMES,model.NODES, initialize=QS_init_rule)  # Reactive power of the SS
    # price init rule
    def PRICE_init_rule(model, time):
        return PRICE[time]

    model.PRICE = Param(model.TIMES, initialize=PRICE_init_rule, mutable=False)
    # Voltage of nodes
    def Voltage_init(model,time, i):
        if model.Tb[i].value == 1.0:
            temp = model.Vnom
            model.V[time,i].fixed = True
        else:
            temp = model.Vnom
            model.V[time,i].fixed = False
        return temp

    model.V = Var(model.TIMES,model.NODES, initialize=Voltage_init)

    # Define Objective Function
    def min_cost_ext_grid(model):
        return sum(sum(model.PS[time, node] * model.PRICE[time] * 1000 for node in model.NODES) for time in model.TIMES)

    model.obj = Objective(rule=min_cost_ext_grid,sense=minimize)

    #soc update rule
    def soc_update_rule(model, time, node):
        if node not in BATTERY_NODES:
            return Constraint.Skip
        if time == model.TIMES.first():
            return (model.SOC[time, node] == model.battery_initial_soc - (
                        model.energy_change[time, node] * 15.0 / 60.0) / model.battery_capacity)
            # return (model.SOC[time, node] == model.battery_initial_soc)
        else:
            return (model.SOC[time, node] == model.SOC[model.TIMES.prev(time), node] - (
                        model.energy_change[time, node] * 15.0 / 60.0) / model.battery_capacity)
            # return (model.SOC[time, node] == model.SOC[model.TIMES.prev(time), node] - (
            #         model.energy_change[model.TIMES.prev(time), node] * 15.0 / 60.0) / model.battery_capacity)

    model.constraint_soc_update=Constraint(model.TIMES,model.NODES,rule=soc_update_rule)

    # for line k consumption == injection
    def active_power_flow_rule(model, time,k):

        return (sum(model.P[time,(j, i)] for j, i in model.LINES if i == k) - sum(
            model.P[time,(i, j)] + model.RM[i, j] * (model.I[time,(i, j)] ** 2) for i, j in model.LINES if k == i) + model.PS[time,k] + model.energy_change[time, k]==
                model.PD[time,k])

    model.active_power_flow = Constraint(model.TIMES,model.NODES, rule=active_power_flow_rule)

    def reactive_power_flow_rule(model,time, k):
        return (sum(model.Q[time,(j, i)] for j, i in model.LINES if i == k) - sum(
            model.Q[time,(i, j)] + model.XM[i, j] * (model.I[time,(i, j)] ** 2) for i, j in model.LINES if k == i) + model.QS[time,k] ==
                model.QD[time,k])
        # return (sum(model.Q[time, (j, i)] for j, i in model.LINES if i == k) - sum(
        #     model.Q[time, (i, j)] for i, j in model.LINES if k == i) +
        #         model.QS[time, k] ==
        #         model.QD[time, k])

    # role of voltage drop
    def voltage_drop_rule(model, time,i,j):
        return ((model.V[time, i] ** 2 - model.V[time, j] ** 2 ) == 2 * (
                    model.RM[i, j] * model.P[time, (i, j)] + model.XM[i, j] * model.Q[time, (i, j)]) + (
                model.RM[i, j] ** 2 + model.XM[i, j] ** 2) * model.I[time, (i, j)] ** 2)
        # return ((model.V[time, i] ** 2 - model.V[time, j] ** 2) == 2 * (
        #         model.RM[i, j] * model.P[time, (i, j)] + model.XM[i, j] * model.Q[time, (i, j)]))

    model.voltage_drop = Constraint(model.TIMES,model.LINES, rule=voltage_drop_rule)

    def define_current_rule(model, time,i, j):
        return ((model.I[time,(i, j)] ** 2) * (model.V[time,i] ** 2) == model.P[time,(i, j)] ** 2 + model.Q[time,(i, j)] ** 2)

    model.define_current = Constraint(model.TIMES,model.LINES, rule=define_current_rule)

    def current_limit_rule(model,time, i, j):
        return (0, model.I[time,(i, j)], None)
    model.current_limit = Constraint(model.TIMES,model.LINES, rule=current_limit_rule)

    def voltage_limit_rule(model, time, i):
        if i in BATTERY_NODES:
            return (model.Vmin, model.V[time, i], model.Vmax)
        return Constraint.Skip

    model.voltage_limit = Constraint(model.TIMES,model.NODES, rule=voltage_limit_rule)

    return model

def convert_dict_to_pd(data:dict):
    df = pd.DataFrame(columns=list(set([k[1] for k in data.keys()])))
    for key, value in data.items():
        df.loc[key[0], key[1]] = value
    return df

def get_pyomo_result(env, year, month, day):
    print("pyomo_year",year)
    print("pyomo_month",month)
    print("pyomo_day",day)
    node_num = env.node_num
    branch_info_file = env.network_info['branch_info_file']
    branch = pd.read_csv(branch_info_file)
    f = np.real(branch.iloc[:, 0]-1).astype(int)  ## list of "from" buses
    t = np.real(branch.iloc[:, 1]-1).astype(int)  ## list of "to" buses
    r = branch.iloc[:, 2]
    x = branch.iloc[:, 3]
    LINES = {(f[i], t[i]) for i in range(len(f))} # list of nodes, real index starting from 1
    sorted_LINES = sorted(LINES)
    R = {(f[i], t[i]): np.real(r[i])/121 for i in range(len(f))}
    X = {(f[i], t[i]): np.real(x[i])/121 for i in range(len(f))}
    NODES = env.net.bus_info.NODES.index.to_list()
    Tb = dict()
    for i, node in enumerate(NODES):
        if i == 0:
            Tb[i] = 1
        else:
            Tb[i] = 0
    TIMES = [i for i in range(96)]

    BATTERY_NODES = env.battery_list

    day_data=env.data_manager.select_day_data(int(year), int(month), int(day))

    active_power = day_data[:, :node_num]
    pv_generation = day_data[:, node_num:node_num+node_num]
    price = day_data[:, -1]
    netload = active_power-pv_generation
    PD_array = netload/1000 # TODO:check if I should /1000, /1000 is for the unit conversion KW-->MW
    PRICE = price/1000 # TODO:check if I should /1000


    Data_Network = {'TIMES': TIMES, 'NODES': NODES, 'LINES': sorted_LINES, 'Tb': Tb, 'PD': PD_array, 'QD': None, 'R': R,
                    'X': X, 'BATTERY_NODES': BATTERY_NODES, 'PRICE': PRICE}


    Vnom = 1.0
    Vmax = 1.05
    Vmin = 0.95
    model = construct_opf_model(Vnom, Vmin, Vmax, Data_Network)

    solver = SolverFactory('ipopt')
    solver.options['constr_viol_tol'] = 1e-6
    solver.options['acceptable_tol'] = 1e-6
    solver.options['dual_inf_tol'] = 1e-6
    solver.solve(model, tee=True,)

    objective_value = model.obj()

    # prepare the results
    voltage_after_control = convert_dict_to_pd(model.V.extract_values())
    active_power = convert_dict_to_pd(model.PD.extract_values())
    ext_grid_active_power = convert_dict_to_pd(model.PS.extract_values())
    ext_grid_reactive_power = convert_dict_to_pd(model.QS.extract_values())
    soc = convert_dict_to_pd(model.SOC.extract_values())
    energy_change = convert_dict_to_pd(model.energy_change.extract_values())

    time_index = Data_Network['TIMES']
    node_index = Data_Network['NODES']
    multi_index = pd.MultiIndex.from_product([time_index, node_index], names=["Time", "Node"])

    data_dict = {
        "Voltage_After_Control": voltage_after_control.stack().reindex(multi_index).values,
        "Active_Power": active_power.stack().reindex(multi_index).values,
        "Ext_Grid_Active_Power": ext_grid_active_power.stack().reindex(multi_index).values,
        "Ext_Grid_Reactive_Power": ext_grid_reactive_power.stack().reindex(multi_index).values,
        "SOC": soc.stack().reindex(multi_index).values,
        "Energy_Change": energy_change.stack().reindex(multi_index).values,
    }

    result_df = pd.DataFrame(data_dict, index=multi_index)
    price_series = pd.Series(PRICE, index=time_index)
    result_df["Price"] = result_df.index.get_level_values("Time").map(price_series)
    result_df = result_df.reset_index()
    result_df["Cost"] = result_df["Ext_Grid_Active_Power"] * result_df["Price"]*1000
    result_df["saved_cost"] = result_df["Energy_Change"] * result_df["Price"]*1000

    return result_df
