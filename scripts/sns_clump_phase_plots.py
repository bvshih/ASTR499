import seaborn as sns
import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import clump_mass_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import SeabornFig2Grid as sfg

rc={'font.size': 14, 'axes.labelsize': 14, 'legend.fontsize': 14, 'legend.title_fontsize': 14,
    'axes.titlesize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14, 'xtick.direction': 'in', 'ytick.direction': 'in'}
sns.set(rc=rc)
sns.set_style("ticks", rc=rc)

def model_colors(model): 
	DM_pal = sns.cubehelix_palette(n_colors=7, reverse=True)
	SF_pal = sns.cubehelix_palette(n_colors=4, start=.5, rot=-.75)

	if model == 'P0':
		pcol = SF_pal[0] 
	elif model == 'P0noBHs':
		pcol = SF_pal[1] 
	elif model == 'GM1':
		pcol = SF_pal[2] 
	elif model == 'GM1noBHs':
		pcol = SF_pal[3] 
	elif model == 'GM2':
		pcol = DM_pal[1]
	elif model == 'GM2noBHs':
		pcol = DM_pal[2]
	elif model == 'GM2SI1':
		pcol = DM_pal[3]
	elif model == 'GM3':
		pcol = DM_pal[4]
	elif model == 'GM3noBHs':
		pcol = DM_pal[5]
	elif model == 'GM3SI1':
		pcol = DM_pal[6]
	return pcol

def scatter_clump_phase(models, output): 
	z = {3840: 0.06, 3456: 0.17, 1536: 1.18, 384: 4.58}	
	fig = plt.figure(figsize=(10, 10))
	pcolor = []
	for i in range(len(models)): 
		dat = clump_mass_function.open_pd_table(models[i], output, 4e-08, 1)
		dat['output'] = np.ones(len(dat['model'])) * int(output)
		pcolor.append(model_colors(models[i]))	
		if(i==0): 
			f_dat = clump_mass_function.open_pd_table(models[i], output, 4e-08, 1)
		else: 
			f_dat = pd.concat([dat, f_dat])

	f_dat = f_dat[f_dat['avg_temp[K]'] != 0 ]
	f_dat = f_dat[f_dat['avg_nHden[cm^-3]'] != 0]	
	f_dat = f_dat[f_dat['clump_r_avg[kpc]'] != 0]
	f_dat['logavgtemp[K]'] = np.log10(f_dat['avg_temp[K]'])
	f_dat['logavg_nHden[cm^-3]'] = np.log10(f_dat['avg_nHden[cm^-3]'])
	f_dat['logSize[kpc]'] = np.log10(f_dat['clump_r_avg[kpc]'])

	min_r = min(f_dat['r[kpc]'])
	max_r = max(f_dat['r[kpc]'])
	scatter = sns.scatterplot(data=f_dat, x='logavg_nHden[cm^-3]', y='logavgtemp[K]', hue='model', s=f_dat['r[kpc]']*2,palette=pcolor)
	#scatter = sns.scatterplot(data=f_dat, x='logavg_nHden[cm^-3]', y='logavgtemp[K]', size='logSize[kpc]', hue='output', sizes=(20, 200), ax=axs[i], legend=False)
	
	#plot points for legend 
	plt.scatter(-9, 4.5, c='k', s=2*10, label='r = ' + str(10) + ' kpc')
	plt.scatter(-9, 4.5, c='k', s=2*100, label='r = ' + str(100) + ' kpc')
	plt.scatter(-9, 4.5, c='k', s=2*200, label='r = ' + str(200) + ' kpc')
	plt.ylabel(r'T [K]')
	plt.annotate('z = ' + str(z[int(output)]), xy=(0.1, 0.9), xycoords='axes fraction')
	
	print(f_dat)
	plt.xlabel(r'n$_{H}$ [cm$^{-3}$]') 
	plt.ylim(3.5, 6.5)
	plt.xlim(-8, 0)
	plt.legend(ncol=2)
	plt.savefig('GMs_scatter_phaseplot_output' + output + '.pdf')
	plt.show()

def joint_clump_phase(model, output, outfile, phasex, phasey, legend, show, g_return): 
	print("models = ", model) 
	print("len of models = ", len(model))
	z = {3840: 0.06, 3456: 0.17, 1536: 1.18, 384: 4.58}	
	pcolor = [] 
		
	for i in range(len(model)):
		print("model = ", model[i])
		print("ouput = ", output)
		dat = clump_mass_function.open_pd_table(model[i], output, 4e-08, 1)
		dat['output'] = np.ones(len(dat['model'])) * int(output)
		
		if len(dat) > 0: 
			pcolor.append(model_colors(model[i]))

		if(i==0): 
			f_dat = clump_mass_function.open_pd_table(model[i], output, 4e-08, 1)
		else: 
			f_dat = pd.concat([dat, f_dat])

		f_dat = f_dat[f_dat['avg_temp[K]'] != 0 ]
		f_dat = f_dat[f_dat['avg_nHden[cm^-3]'] != 0]	
		f_dat = f_dat[f_dat['clump_r_avg[kpc]'] != 0]

		f_dat['logavgtemp[K]'] = np.log10(f_dat['avg_temp[K]'])
		f_dat['logavg_nHden[cm^-3]'] = np.log10(f_dat['avg_nHden[cm^-3]'])
		f_dat['logSize[kpc]'] = np.log10(f_dat['clump_r_avg[kpc]'])
		f_dat['logvdist'] = np.log10(f_dat['vdist[km/s]'])
		f_dat['logpressure'] = np.log10(f_dat['avg_pressure[Pa]'])
		f_dat = f_dat[f_dat['clump_r_avg[kpc]'] > 0.25]

		if(phasex == 'nHden'):
			datx = 'logavg_nHden[cm^-3]'
			xlab = r'log$_{10}$ n$_{H}$ [cm$^{-3}$]'
			xlim = [-8, 0]
		elif(phasex == 'vdisp'):
			datx = 'logvdist'
			xlab = r'log$_{10}$ $\sigma_v$ [km/s]'
			xlim = [-1, 2.5]
		elif(phasex == 'pressure'):
			datx = 'logpressure'
			xlab = r'log$_{10}$ Pressure [Pa]'
			xlim = [5.5, 12]	
		elif(phasex == 'temp'):
			datx = 'logavgtemp[K]'
			xlab = r'log$_{10}$ T [K]'
			xlim = [3.5, 6.5]
		
		if(phasey == 'temp'):
			daty = 'logavgtemp[K]'
			ylab = r'log$_{10}$ T [K]'
			ylim = [3.5, 6.5]
		elif(phasey == 'pressure'): 
			daty = 'logpressure'   
			ylab = r'log$_{10}$ Pressure [Pa]'
			ylim = [5.5, 12]	

	#print(f_dat['model'])
	#print(np.unique(f_dat['model']))
	#print(len(np.unique(f_dat['model'])))
	num_models = len(np.unique(f_dat['model']))

	pcolor_r = []
	for i in range(len(pcolor)): 
		pcolor_r.append(pcolor[len(pcolor)-1-i])
	g = sns.jointplot(data=f_dat, x=datx, y=daty, hue='model', s=2*f_dat['r[kpc]'], palette=pcolor_r, height=10, legend=True)
	
	if legend==1:
		#plot points for legend 
		g.ax_joint.scatter(-9, 4.5, c='k', s=2*10, label='r = ' + str(10) + ' kpc')
		g.ax_joint.scatter(-9, 4.5, c='k', s=2*100, label='r = ' + str(100) + ' kpc')
		g.ax_joint.scatter(-9, 4.5, c='k', s=2*200, label='r = ' + str(200) + ' kpc')
		g.set_axis_labels(xlab, ylab)
		g.ax_joint.annotate(r'z = ' + str(z[int(output)]), xy=(0.1, 0.9), xycoords='axes fraction', fontsize=14)
	
	#temp
	g.ax_marg_y.set_ylim(ylim[0], ylim[1])

	#vdisp
	g.ax_marg_x.set_xlim(xlim[0], xlim[1])

	#nden
	#g.ax_marg_x.set_xlim(-8, 0)
	#g.ax_joint.legend(ncol=2)

	#plt.savefig(outfile)
	
	if show:
		plt.show()	
	else: 
		plt.close()

	if g_return: 
		return g

def grid_phase(model_type, models, outputs, phasex, phasey, show): 
	outfile = './plots/model_cats_' + phasex + '_' + phasey + '.pdf'
	#fig = plt.figure(figsize=(26,16))
	#fig = plt.figure(figsize=(8,8))
	fig = plt.figure(figsize=(len(model_type)*4, len(outputs)*4))
	#fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True, figsize=(8,8))
	gs = gridspec.GridSpec(len(model_type), len(outputs))

	#create array that is all zeros except for the last index to put a legend on the very last column
	legend = np.append(np.zeros(len(outputs)-1),1)
	g = [] 
	
	for l in range(len(model_type)): 
			

		print('model category = ', model_type[l])
		print('number of models in category = ', len(models[l]))		
		#print('number of models in category ' + str(l+1) + ' is ' + str(len(models_cat)))

		for m in range(len(outputs)):
			g.append(joint_clump_phase(models[l], outputs[m], outfile, phasex, phasey, legend[m], False, True))

	for k in range(len(g)):
		mg = sfg.SeabornFig2Grid(g[k], fig, gs[k])
	gs.tight_layout(fig)
	plt.savefig(outfile, dpi=300, bbox_inches='tight')

	#g.ax_joint.legend(ncol=2)
	if show:
		plt.show()	
	else: 
		plt.close()

	
	
	
