#finds the min and max axes limits 

import sns_clump_phase_plots
import clump_mass_function
import numpy as np
import pandas as pd

models = ['P0', 'GM1', 'GM2', 'GM3', 'P0noBHs', 'GM1noBHs', 'GM2noBHs', 'GM3noBHs', 'GM2SI1', 'GM3SI1' ]
outputs = ['000384', '001536', '003456']

qty = ['logavgtemp[K]', 'logavg_nHden[cm^-3]', 'logsize[kpc]', 'logvdist', 'logpressure','logmass']

outfn = '/scratch/08263/tg875625/ASTR499/scripts/datfiles/phase_axis_limits.dat'

with open(outfn, 'w') as outfile:
	outfile.write('qty min max\n')

for i in range(len(models)):
    for j in range(len(outputs)):
        print("model = ", models[i])
        print("ouput = ", outputs[j])
        dat = clump_mass_function.open_pd_table(models[i], outputs[j], 4e-08, 1)
        dat['output'] = np.ones(len(dat['model'])) * int(outputs[j])

        if(i==0): 
            f_dat = clump_mass_function.open_pd_table(models[i], outputs[i], 4e-08, 1)
        else: 
            f_dat = pd.concat([dat, f_dat])

        f_dat = f_dat[f_dat['avg_temp[K]'] != 0 ]
        f_dat = f_dat[f_dat['avg_nHden[cm^-3]'] != 0]	
        f_dat = f_dat[f_dat['clump_r_avg[kpc]'] != 0]

        f_dat['logavgtemp[K]'] = np.log10(f_dat['avg_temp[K]'])
        f_dat['logavg_nHden[cm^-3]'] = np.log10(f_dat['avg_nHden[cm^-3]'])
        f_dat['logsize[kpc]'] = np.log10(f_dat['clump_r_avg[kpc]'])
        f_dat['logvdist'] = np.log10(f_dat['vdist[km/s]'])
        f_dat['logpressure'] = np.log10(f_dat['avg_pressure[Pa]'])
        f_dat['logmass'] = np.log10(f_dat['clump_mass[Msol]'])

        #filters to only include CGM
        f_dat = f_dat[f_dat['clump_r_avg[kpc]'] > 0.25]
for i in qty:
    limits = [f_dat[i].min(), f_dat[i].max()]
    with open(outfn, 'a') as outfile: 
        outfile.write('%s %f %f \n' %(i, limits[0], limits[1]))