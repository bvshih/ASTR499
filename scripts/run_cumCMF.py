import clump_mass_function

#models = ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs', 'GM2SI1', 'GM3SI1' ]

models  = ['P0']
#models = ['GM1', 'GM2']
outputs = ['000384', '001536', '003456']

clump_mass_function.cum_CMF(models, outputs,  4e-08, 1)