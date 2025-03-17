# from pyomo.environ import *

# # Simple test model
# model = ConcreteModel()
# model.x = Var(bounds=(0, 10), initialize=5)
# model.obj = Objective(expr=model.x**2, sense=minimize)
# model.con = Constraint(expr=model.x >= 3)

# # Solve the model
# from pyomo.opt import SolverFactory
# opt = SolverFactory('gurobi')
# results = opt.solve(model)

# # Display results
# model.display()

from pyomo.opt import SolverFactory
from pyomo.environ import *

# Set Solver Path explicitly if needed
opt = SolverFactory('gurobi', executable='C:/gurobi1200/win64/bin/gurobi_cl')

if opt is None:
    print("Pyomo cannot find the 'gurobi' solver.")
else:
    print("Pyomo successfully found the 'gurobi' solver.")

