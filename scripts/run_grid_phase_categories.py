import sns_clump_phase_plots
#sns_clump_phase_plots.grid_phase([['P0', 'GM1', 'GM2', 'GM3'], ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs'], ['GM2', 'GM2SI1', 'GM3', 'GM3SI1']], ['000384', '001536', '003456'], 'nHden', 'temp', True)

model_type = ['SFvsQ', 'BHnoBH', 'DM']
model_cat = [['P0', 'GM1', 'GM2', 'GM3'], ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs'], ['GM2', 'GM2SI1', 'GM3', 'GM3SI1']] 
outputs = ['000384', '001536', '003456']

sns_clump_phase_plots.grid_phase(model_type, model_cat, outputs, 'nHden', 'temp', False)

#for k in range(len(model_cat)):
#	for l in range(len(outputs)):
#		print("model_cat[k] = ", model_cat[k])
#		sns_clump_phase_plots.grid_phase(model_cat[k], outputs[l], 'nHden', 'temp', False)