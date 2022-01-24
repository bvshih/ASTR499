import sns_clump_phase_plots
#sns_clump_phase_plots.grid_phase([['P0', 'GM1', 'GM2', 'GM3'], ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs'], ['GM2', 'GM2SI1', 'GM3', 'GM3SI1']], ['000384', '001536', '003456'], 'nHden', 'temp', True)

model_type = ['SFvsQ', 'BHnoBH', 'DM']
model_cat = [['P0', 'GM1', 'GM2', 'GM3'], ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs'], ['GM2', 'GM2SI1', 'GM3', 'GM3SI1']] 
outputs = ['000384', '001536', '003456']

phase = [['nHden','temp'], ['size', 'mass'], ['size', 'vdist']]

#no legends
legend = True

for i in range(len(phase)):
    print(phase[i][0] + " vs " + phase[i][1])
    sns_clump_phase_plots.grid_phase(model_type, model_cat, outputs, phase[i][0], phase[i][1], legend, False)
