from cmath import nan
from distutils.command.build_scripts import first_line_re
from distutils.log import error
from multiprocessing.sharedctypes import Value
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib.cm as cm 
import sns_clump_phase_plots
import random
import itertools
from scipy.optimize import curve_fit

def open_pd_table(model, output, HI_cut, link_len):
	
	dat_directory = '/scratch/08263/tg875625/ASTR499/scripts/datfiles/' #bevs directory 
	#dat_directory = '/scratch/06040/adcruz/pyscripts/CGM_SIDM/scripts/cold_clumps2/datfiles/'
	pd_fn = model + '_' + output + '_HI' + str(HI_cut) + '_' + str(link_len) + 'kpc_clump_data.dat'
	
	pd_table = pd.read_table(dat_directory + pd_fn, sep = '\s+')
	return pd_table 
 
def pdtable_put_output(pd_table, output):
	output_array = int(output) * np.ones(len(pd_table))
	pd_table['output'] = output_array
	
	return pd_table 

def pdtable_put_z(pd_table, output, z):
	z_array = float(z) * np.ones(len(pd_table))
	pd_table['z'] = z_array
	
	return pd_table 

def pdtable_put_model(pd_table, model): 
	output_array = [] 
	for i in range(len(pd_table)): 
		output_array.append(model)
 
	pd_table['model'] = output_array 
	
	return pd_table 

def pdtable_put_model_z(pd_table, model, output):
	z = {3840: 0.06, 3456: 0.17, 1536: 1.18, 384: 4.58}	
	model_z_array = []
	for i in range(len(pd_table)):
		model_z_array.append(model + ' ' + str(z[output]))
	pd_table['model z'] = model_z_array

	return pd_table

def pdtable_fix_CM_units(model, output, HI, fof_grp, dispersion, HI_cutfix):
	dSolUnit = 1.5928853e16

	#pd_table = pd.read_table(model + '_' + output + '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_fof_clumps.dat', sep = '\s+')
	pd_table = open_pd_table(model, output, HI, fof_grp, dispersion, HI_cutfix)
	pd_table = pd_table[1:]

	pd_table['clump_mass[M_simUnits]'] = pd_table['clump_mass[Msol]'].astype(float)
	pd_table['clump_mass[Msol]']        = pd_table['clump_mass[M_simUnits]'] / dSolUnit

	return pd_table

def combine_steps(model, outputs, z, HI, link_len, putz):
	for i in range(len(outputs)):
		#pd_table = pd.read_table(model + '_' + outputs[i] + '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_fof_clumps.dat', sep = '\s+')
		pd_table = open_pd_table(model, outputs[i], HI, link_len)

		pd_table = pd_table[1:]

		if(putz == True):
			pd_table = pdtable_put_z(pd_table, outputs[i], z[i])
		else:
			pd_table = pd_table

		if(i == 0):
			pd_tables = pd_table
		else:
			pd_tables = pd.concat([pd_tables, pd_table])
	return pd_tables

def combine_sims(models, output, z, HI, link_len):
	for i in range(len(models)):
		pd_table = open_pd_table(models[i], output, HI, link_len)

		pd_table = pd_table[1:]
	 
		pd_table = pdtable_put_model(pd_table, models[i])
	
		if(i == 0):
			pd_tables = pd_table
		else:
			pd_tables = pd.concat([pd_tables, pd_table])
	return pd_tables

	
def get_logbins(pd_table): 
	#pd_table = combine_steps(model, outputs, z, HI, link_len, putoutput)

	#get log bins 
	min_CM = min(pd_table['clump_mass[Msol]'])
	max_CM = max(pd_table[pd_table['clump_mass[Msol]'] < 10**10]['clump_mass[Msol]'])
	log_bins = np.logspace(np.log10(min_CM),np.log10(max_CM), 40)
	
	return log_bins	

def sing_sns_stack_CMhist(model, outputs, HI, fof_grp, fixUnits, putoutput, dispersion, HI_cutfix): 
	pd_table = combine_steps(model, outputs, HI, fof_grp, putoutput, dispersion, HI_cutfix)
	pd_table['log_clump_mass[Msol]'] = np.log10(pd_table['clump_mass[Msol]'])

	#get log bins 
	min_CM = min(pd_table['clump_mass[Msol]'])
	max_CM = max(pd_table[pd_table['clump_mass[Msol]'] < 10**10]['clump_mass[Msol]'])
	log_bins = np.logspace(np.log10(min_CM),np.log10(max_CM), 40)

	f, ax = plt.subplots(figsize=(12, 10))

	sns.histplot(
	pd_table,
	x = "log_clump_mass[Msol]", hue = "output",
    	multiple="stack",
    	palette="light:m_r",
    	edgecolor=".3",
    	linewidth=.5,
    	bins = np.log10(log_bins), 
	ax = ax
	)

	ax.set_xlabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 18)
	ax.set_ylabel('Count', fontsize = 18)
	ax.tick_params(labelsize = 18)
	plt.show()

def pair_sns_stack_CMhist(models, outputs, z, HI, link_len, putz, hist_type): 
	plotpath = '/scratch/08263/tg875625/plots/'

	f, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(20, 10), sharey = True, sharex=True)
	#f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 10), sharey = True)
	colors = [ '#542BC4', '#7649CD', '#BB84E0', '#FFBFF3']
	snspalette = sns.cubehelix_palette(n_colors=4, reverse=True)
	axs = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
	for m in range(len(models)): 

		pd_table = combine_steps(models[m], outputs, z, HI, link_len, putz)
		pd_table['log_clump_mass[Msol]'] = np.log10(pd_table['clump_mass[Msol]'])
		if(models[m] == 'GM3'): 
			snspalettef = snspalette[:-1]
		else: 
			snspalettef = snspalette

		#get log bins 
		#log_bins = get_logbins(bin_model, outputs, z, HI, link_len, putz)
		log_bins = get_logbins(pd_table)
		
		if(hist_type == 'stack'):
			sns.histplot(
			pd_table,
			x = "log_clump_mass[Msol]", hue = "z",
    			multiple = "stack",
    			palette = snspalettef,
    			edgecolor = ".3",
    			linewidth = 0.5,
    			bins = np.log10(log_bins), 
			ax = axs[m]
			)
		elif(hist_type == 'hue'): 
			sns.histplot(
			pd_table,
			x = "log_clump_mass[Msol]", hue = "z",
    			palette = snspalettef,
    			edgecolor = ".3",
    			linewidth = 0.5,
    			bins = np.log10(log_bins), 
			ax = axs[m]
			)
		elif(hist_type == 'step'):
			sns.histplot(
			pd_table,
			x="log_clump_mass[Msol]", hue = "z",
    			palette = snspalettef,
    			edgecolor = ".3",
    			linewidth = 0.5,
    			element = 'step',
			bins = np.log10(log_bins), 
			ax = axs[m]
			)
		elif(hist_type == 'dodge'):
			sns.histplot(
			pd_table,
			x="log_clump_mass[Msol]", hue = "z",
    			palette = snspalettef,
    			edgecolor = ".3",
    			linewidth = 0.5,
    			multiple='dodge',
			bins = np.log10(log_bins), 
			ax = axs[m]
			)
		else: 
			print('pick hist_type = dodge, hue, stack or step')

		axs[m].set_xlabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 16)
		axs[m].set_ylabel('log$_{10}$ Count', fontsize = 16)
		axs[m].tick_params(labelsize = 16)
		axs[m].annotate(models[m], xy = (8, 5), fontsize = 16)
		#AC test loglog 8/24/21 
		axs[m].semilogy()
	
	#plt.savefig(plotpath + hist_type + '_' + models[0] + '_suite_' + outputs[0] + '-'+ outputs[len(outputs)-1] +  '_' + 'HI' + str(HI) + '_' + str(link_len) + 'kpc_clump_hist.pdf')
	plt.savefig(plotpath + hist_type + '_' + models[0] + models[1] + models[2] + models[3] + '_suite_' + outputs[0] + '-'+ outputs[len(outputs)-1] +  '_' + 'HI' + str(HI) + '_' + str(link_len) + 'kpc_clump_hist.pdf')
	plt.show()

def cum_CMF(models, outputs, HI, link_len):
	z = {3840: 0.06, 3456: 0.17, 1536: 1.18, 384: 4.58}	
	fig, ax = plt.subplots(2, len(outputs), figsize=(len(outputs)*10, 20))
	#linestyle = ['-', '--']
	#custom_lines = [Line2D([0], [0], linestyle='-', color='k', linewidth=3), Line2D([0], [0], linestyle='--', color='k', linewidth=3)]
	colors = []
			
	#4 breaks per z 
	#guess break values
	mass_guess = [[[10**5.2, 10**5.3],[10**5.7, 10**6],[10**6.1, 10**6.3]], #z1
				  [[10**5.2, 10**5.4],[10**6.4,10**6.7],[10**7.3, 10**7.8]], #z2
				  [[10**5.2, 10**5.4],[10**6.4, 10**6.7,],[10**6.9, 10**7]]] # z3

	#intitial guess for curve fit [m,b]
	p0 = [[[0, 600], [-10, 600],[-100, 600], [-1, 10]],
		  [[-10, 10**3], [-10**3, 10**4],[-10**4, 10**3], [-10**3, 10**3]],
		  [[-10, 10**2], [-10**3, 10**3],[-10**4, 10**3], [-10**3, 10**3]]]
	
	iterations = 10

	for i in range(len(models)):
		colors.append(sns_clump_phase_plots.model_colors(models[i]))
		print('added color for ' + models[i])

	for i in range(len(models)): 
		print('model = ' + models[i])
		for j in range(len(outputs)):
			print('z = ' +  str(outputs[j]))
			pd_table = open_pd_table(models[i], outputs[j], HI, link_len)
			
			mass_mask = pd_table['clump_mass[Msol]'] < 10**10
			mass = pd_table[mass_mask]['clump_mass[Msol]']
			mass = np.flip(np.sort(mass))

			mass_counts = {}
			for m in mass:
				N = mass[mass>m].shape[0]
				mass_counts[m] = N
			
			fitted_counts = find_fit(mass_counts, mass_guess[j], p0[j], iterations)

			dN = calc_dN(np.array(list(mass_counts.keys()), dtype=float), np.array(list(mass_counts.values()),dtype=int))
			fitted_dN = calc_dN(np.array(list(fitted_counts.keys()), dtype=float), np.array(list(fitted_counts.values()),dtype=int))

			ax[0,j].scatter(mass_counts.keys(), mass_counts.values(), color = colors[i], label = models[i], alpha = 0.5)	
			ax[0,j].plot(fitted_counts.keys(),fitted_counts.values(), linestyle='-', linewidth = 3, label = models[i])

			ax[1,j].scatter(list(mass_counts.keys())[1:], dN, color = colors[i], label = models[i], alpha = 0.5)	
			ax[1,j].plot(list(fitted_counts.keys())[1:], fitted_dN, linestyle='-', linewidth = 3, color = colors[i], label = models[i])

			ax[0,j].set_ylim(1, 5e3) 
			ax[0,j].set_xlim(1e5, 4e9)
			ax[0,j].set_xscale('symlog')
			ax[0,j].set_yscale('symlog')
			ax[0,j].tick_params(labelsize = 18, direction='in')
			ax[0,j].tick_params(which='minor', direction='in')

			ax[1,j].set_ylim(-1e9, 0) 
			ax[1,j].set_xlim(1e5, 4e9)
			ax[1,j].set_xscale('symlog')
			ax[1,j].set_yscale('symlog')
			ax[1,j].tick_params(labelsize = 18, direction='in')
			ax[1,j].tick_params(which = 'minor', direction='in')

			ax[0,j].set_xlabel(r'log$_{10}$ M$_{\mathrm{clump}}$ [M$_{\odot}$]', fontsize = 18)
			ax[1,j].set_xlabel(r'log$_{10}$ M$_{\mathrm{clump}}$ [M$_{\odot}$]', fontsize = 18)

	ax[0,0].set_ylabel(r'N( > M$_{\mathrm{clump}}$)', fontsize = 18)
	ax[1,0].set_ylabel(r'$\frac{dN}{dlogM}$', fontsize = 18)
	#legend1 = ax[0,len(outputs)-1].legend(fontsize = 14, ncol = 1)  
	#ax.legend(custom_lines, models, loc='lower left', fontsize=14)
	#plt.gca().add_artist(legend1)
	plt.savefig('/scratch/08263/tg875625/ASTR499/scripts/plots/P0_curvefit_cumCMF_HI' + str(HI) + '_' + str(link_len) + 'kpc.pdf')
	

def clump_size_relation(models, outputs, z, HI_cut, subplot, qty, link_len):
	if(subplot == True): 
		f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 10), sharey = True, sharex = True)
	else:
		fig = plt.figure(figsize=(12, 10))
	markers = ['o', 's']
	alphas  = [1, 0.5]

	evenly_spaced_interval = np.linspace(0.1, 1, len(outputs))
	if(qty == 'mass'):
		colors = [cm.PuRd(x) for x in evenly_spaced_interval]
	elif(qty == 'temp'): 
		colors = [cm.Blues(x) for x in evenly_spaced_interval]
	elif(qty == 'pressure'):
		colors = [cm.twilight(x) for x in evenly_spaced_interval]
	elif(qty == 'density'):
		colors = [cm.cividis(x) for x in evenly_spaced_interval]
	elif(qty == 'vdist'):  	
		colors = [cm.summer(x) for x in evenly_spaced_interval]
	else: 
		print('havent picked avail option')
	
	for i in range(len(models)): 
		for j in range(len(outputs)):	
			
			#print('model, output: ' + models[i] + ', ' + outputs[j])
			#change to read in clump.dat
			pd_table = open_pd_table(models[i], outputs[j], HI_cut, link_len)
			#make sure these match the column titles in the dat file
			mass_mask = pd_table['clump_mass[Msol]'] < 10**10
			size = pd_table[mass_mask]['clump_r_avg[kpc]'] 
			if(qty == 'mass'):
				p_qty = pd_table[mass_mask]['clump_mass[Msol]']
			elif(qty == 'temp'): 
				p_qty = pd_table[mass_mask]['avg_temp[K]']
			elif(qty == 'pressure'):
				p_qty = pd_table[mass_mask]['avg_pressure[Pa]']
			elif(qty == 'density'): 
				p_qty = pd_table[mass_mask]['avg_nHden[cm^-3]'].astype(float)
				#print(str(min(np.log10(p_qty))))
			elif(qty == 'vdist'):
				p_qty = pd_table[mass_mask]['vdist[km/s]']
			else: 
				print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 
	
			if(subplot == True):
				ax[i].scatter(size, p_qty, marker = markers[i], s = 100, label = models[i] + ' z = ' + '%0.2f'%(z[j]), color = colors[j], edgecolor="white")
			else: 
				plt.scatter(size, p_qty, marker = markers[i], s = 100, alpha = alphas[i], label = models[i] + ' z = ' + '%0.2f'%(z[j]), color = colors[j], edgecolor="white")
	if(subplot == True): 
		for i in range(len(models)):
			if(qty == 'mass'):
				ax[i].set_ylabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 18)
			elif(qty == 'temp'): 
				ax[i].set_ylabel(r'log$_{10}$ Temperature [K]', fontsize = 18)
			elif(qty == 'pressure'): 
				ax[i].set_ylabel(r'log$_{10}$ Pressure [Pa]', fontsize = 18)
			elif(qty == 'density'): 
				ax[i].set_ylabel(r'log$_{10}$ n$_H$ [cm$^{-3}$]', fontsize = 18)	
			elif(qty == 'vdist'): 
				ax[i].set_ylabel(r'$\sigma_v$ [km/s]', fontsize = 18)
			else: 
				print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 
				
			ax[i].set_xlabel(r'size [kpc]', fontsize = 18)
			ax[i].tick_params(labelsize = 18)
			ax[i].loglog()
			if(qty == 'density'): 
				ax[i].set_ylim(1e-7, 1e-1)
			elif(qty == 'vdist'): 
				ax[i].set_ylim(4e-1, 2e2)	
			ax[i].legend(fontsize = 14, ncol = 2)
		plt.savefig('/scratch/08263/tg875625/plots/' + models[0] + '_' + models[1] + '_' + outputs[0] + '-'+ outputs[len(outputs)-1] +  '_' + 'HI' + str(HI_cut) + '_'  + str(link_len) + 'kpc_clump_size_' + qty + '_relation.pdf')
	else:
		if(qty == 'mass'):
			plt.ylabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 18)
		elif(qty == 'temp'): 
			plt.ylabel(r'log$_{10}$ Temperature [K]', fontsize = 18)
		elif(qty == 'pressure'): 
			plt.ylabel(r'log$_{10}$ Pressure [Pa]', fontsize = 18)
		elif(qty == 'density'): 
			plt.ylabel(r'log$_{10}$ n$_H$ [cm$^{-3}$]', fontsize = 18)	
		elif(qty == 'vdist'): 
			plt.ylabel(r'$\sigma_v$ [km/s]', fontsize = 18)
		else: 
			print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 


		plt.xlabel(r'size [kpc]', fontsize = 18)
		plt.tick_params(labelsize = 18)
		plt.loglog() 
		plt.legend(fontsize = 18, ncol = 2)
		plt.savefig('/scratch/08263/tg875625/plots/' + models[0] + '_' + outputs[0] + '-'+ outputs[len(outputs)-1] +  '_' + 'HI' + str(HI_cut) + '_' + str(link_len) + 'kpc_clump_size_' + qty + '_relation.pdf')
	plt.show()

def clump_r_phi_pos(models, outputs, HI, fof_grp, dispersion, HI_cutfix):
	f, ax = plt.subplots(nrows = len(models), ncols = len(outputs), figsize=(20, 10), sharey = True, sharex = True)
	
	markers = ['o', 's']

	for i in range(len(models)): 
		for j in range(len(outputs)):	
			pd_table = open_pd_table(models[i], outputs[j], HI, fof_grp, dispersion, HI_cutfix)
		
			if(dispersion == False):
				pd_table = pdtable_fix_CM_units(models[i], outputs[j], HI, fof_grp, dispersion, HI_cutfix)
			else:
				pd_table = pd_table

			mass_mask = pd_table['clump_mass[Msol]'] < 10**10
			mass = pd_table[mass_mask]['clump_mass[Msol]']
			r    = pd_table[mass_mask]['r[kpc]']
			phi  = pd_table[mass_mask]['phi[degree]'] 
			phi  = ((phi * (np.pi / 180)) + (np.pi / 2) ) * (180/np.pi)
			im = ax[i][j].scatter(r, phi, c = np.log10(mass), marker = markers[i],s=np.log10(mass) * 15, edgecolor='white', cmap = 'viridis')
			ax[i][j].set_title(models[i] + ' ' + outputs[j])
			f.colorbar(im, ax = ax[i][j])
			im.set_clim(6, 8.6) 		
	
			if(j == 0 ):
				ax[i][j].set_ylabel(r'$\phi$[degree]', fontsize = 15)
				if(i == 1):	
					ax[i][j].set_xlabel('r[kpc]', fontsize = 15)
			elif(i == 1): 	
				ax[i][j].set_xlabel('r[kpc]', fontsize = 15)
			ax[i][j].tick_params(labelsize = 15) 
	
	plt.savefig(models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_r_vs_phi.pdf')
	plt.show()
			
def clump_rpos_mass(models, outputs, redshift, HI, fof_grp, dispersion, HI_cutfix):
	f, ax = plt.subplots(nrows = 1, ncols = len(outputs), figsize=(16, 6), sharey = True, sharex = True)
	
	markers = ['o', 's']
	colors  = ['palevioletred', 'cornflowerblue']
	for i in range(len(models)): 
		for j in range(len(outputs)):	
			pd_table = open_pd_table(models[i], outputs[j], HI, fof_grp, dispersion, HI_cutfix)
		
			if(dispersion == False):
				pd_table = pdtable_fix_CM_units(models[i], outputs[j], HI, fof_grp, dispersion, HI_cutfix)
			else:
				pd_table = pd_table

			mass_mask = pd_table['clump_mass[Msol]'] < 10**10
			mass = pd_table[mass_mask]['clump_mass[Msol]']
			r    = pd_table[mass_mask]['r[kpc]']
			phi  = pd_table[mass_mask]['phi[degree]'] 
			phi  = ((phi * (np.pi / 180)) + (np.pi / 2) ) * (180/np.pi)
			im = ax[j].scatter(r, mass, marker = markers[i],s=np.log10(mass) * 30, edgecolor='white', color=colors[i], label = models[i])
			#ax[j].set_title(models[i] + ' ' + outputs[j])
			ax[j].text(10**(1.82), 10**(8.5), 'z = ' + '%0.2f'%(redshift[j]), fontsize = 15)
			if(j == 0 ):
				ax[j].set_ylabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 15)
			elif(i == 1): 	
				ax[j].set_xlabel('log$_{10}$ r[kpc]', fontsize = 15)
			ax[j].tick_params(labelsize = 15)
			ax[j].loglog() 
		ax[0].legend(fontsize = 15, loc = 'upper left')
	if(HI_cutfix == True): 
		plt.savefig('HI_cutfix_' + models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_rpos_vs_mass_tog.pdf')
	else:
		plt.savefig(models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_rpos_vs_mass_tog.pdf')
	plt.show()

def clump_phase(models, outputs, z, HI, fof_grp, dispersion, subplot, qty, HI_cutfix):
	if(subplot == True): 
		f, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 10), sharey = True, sharex = True)
	else:
		fig = plt.figure(figsize=(12, 10))
	markers = ['o', 's']
	alphas  = [1, 0.5]

	evenly_spaced_interval = np.linspace(0.1, 1, len(outputs))
	if(qty == 'mass'):
		colors = [cm.PuRd(x) for x in evenly_spaced_interval]
	elif(qty == 'temp'): 
		colors = [cm.viridis(x) for x in evenly_spaced_interval]
	elif(qty == 'pressure'):
		colors = [cm.twilight(x) for x in evenly_spaced_interval]
	elif(qty == 'density'):
		colors = [cm.cividis(x) for x in evenly_spaced_interval]
	elif(qty == 'vdist'):  	
		colors = [cm.summer(x) for x in evenly_spaced_interval]
	else: 
		print('havent picked avail option')
	
	for i in range(len(models)): 
		for j in range(len(outputs)):	
			pd_table = open_pd_table(models[i], outputs[j], HI, fof_grp)
		
			#if(dispersion == False):
			#	pd_table = pdtable_fix_CM_units(models[i], outputs[j], HI, fof_grp, dispersion, HI_cutfix)
			#else:
			#	pd_table = pd_table

			mass_mask = pd_table['clump_mass[Msol]'] < 10**10
			den = pd_table[mass_mask]['avg_nHden[cm^-3]'] 
			print(min(den))
			if(qty == 'mass'):
				p_qty = pd_table[mass_mask]['clump_mass[Msol]']
			elif(qty == 'temp'): 
				p_qty = pd_table[mass_mask]['avg_temp[K]']
			elif(qty == 'pressure'):
				p_qty = pd_table[mass_mask]['avg_pressure[Pa]']
			elif(qty == 'density'): 
				p_qty = pd_table[mass_mask]['avg_nHden[cm^-3]']
				print(str(np.log10(p_qty)))
			elif(qty == 'vdist'):
				p_qty = pd_table[mass_mask]['vdist[km/s]']
			else: 
				print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 
	
			if(subplot == True):
				ax[i].scatter(den, p_qty, marker = markers[i], s =100, label = models[i] + ' z = ' + '%0.2f'%(z[j]), color = colors[j], edgecolor="white")
			else: 
				plt.scatter(den, p_qty, marker = markers[i], s =100, alpha = alphas[i], label = models[i] + 'z = ' + '%0.2f'%(z[j]), color = colors[j], edgecolor="white")
	if(subplot == True): 
		for i in range(len(models)):
			if(qty == 'mass'):
				ax[i].set_ylabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 18)
			elif(qty == 'temp'): 
				ax[i].set_ylabel(r'log$_{10}$ Temperature [K]', fontsize = 18)
			elif(qty == 'pressure'): 
				ax[i].set_ylabel(r'log$_{10}$ Pressure [Pa]', fontsize = 18)
			elif(qty == 'density'): 
				ax[i].set_ylabel(r'log$_{10}$ n$_H$ [cm$^{-3}$]', fontsize = 18)	
			elif(qty == 'vdist'): 
				ax[i].set_ylabel(r'$\sigma_v$ [km/s]', fontsize = 18)
			else: 
				print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 
				
			ax[i].set_xlabel(r'N$_H$ [cm$^{-3}$]', fontsize = 18)
			ax[i].tick_params(labelsize = 18)
			ax[i].loglog()
			ax[i].set_xlim(10**(-4.1), 1e-1)
			if(qty == 'density'): 
				ax[i].set_ylim(10**(-4.1), 1e-1)
			ax[i].legend(fontsize = 14, ncol = 1)
		if(HI_cutfix == True): 
			plt.savefig('HI_cutfix_subplots_' + models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_clump_den_' + qty + '_relation.pdf')
		else:
			plt.savefig('subplots_' + models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_clump_den_' + qty + '_relation.pdf')
	else:
		if(qty == 'mass'):
			plt.ylabel(r'log$_{10}$ Clump Mass [M$_{\odot}$]', fontsize = 18)
		elif(qty == 'temp'): 
			plt.ylabel(r'log$_{10}$ Temperature [K]', fontsize = 18)
		elif(qty == 'pressure'): 
			plt.ylabel(r'log$_{10}$ Pressure [Pa]', fontsize = 18)
		elif(qty == 'density'): 
			plt.ylabel(r'log$_{10}$ n$_H$ [cm$^{-3}$]', fontsize = 18)	
		elif(qty == 'vdist'): 
			plt.ylabel(r'$\sigma_v$ [km/s]', fontsize = 18)
		else: 
			print('you havent selected an avail option. pick: mass, temp, pressure, density or vdist') 


		plt.xlabel(r'N$_H$[cm$^-3$]', fontsize = 18)
		plt.tick_params(labelsize = 18)
		plt.loglog() 
		plt.legend(fontsize = 18, ncol = 2)
		plt.savefig(models[0] + 'suite_' + outputs[0] + '-'+ outputs[-1] +  '_' + 'HI' + HI + '_e_m_' + fof_grp + '.grp_clump_size_' + qty + '_relation.pdf')
	plt.show()

def clump_rpos_mass_whist(models, outputs, redshift, HI, fof_grp, dispersion, HI_cutfix): 
	fig = plt.figure(figsize=(12, 10))

	for j in range(len(outputs)):	
	
		pd_table = combine_sims(models, outputs[j], HI, fof_grp, dispersion, HI_cutfix)

		mass_mask = pd_table['clump_mass[Msol]'] < 10**10
		pd_table = pd_table[mass_mask] 

		sns.jointplot(data = pd_table, x = 'r[kpc]', y = 'clump_mass[Msol]')
	
def calc_dN(mass, N):
	log_mass = np.log10(mass)
	derivative = []
	for i in range(len(N)-1):
		derivative.append((N[i+1]-N[i])/(log_mass[i+1]-log_mass[i]))
	#print(derivative)
	return derivative

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def loglog(x,m,b):
	return x**m * b

def calc_fit(data, guess, p0):
	if guess[0] == 0:
		subsection = {k:v for (k,v) in data.items() if k <= guess[1]}
		x = np.array(list(subsection.keys()))
		y = np.array(list(subsection.values()))
	elif guess[0] == 1:
		subsection = {k:v for (k,v) in data.items() if k >= guess[1]}
		subsection = {k:v for (k,v) in subsection.items() if k <= guess[2]}
		x = np.array(list(subsection.keys()))
		y = np.array(list(subsection.values()))
	elif guess[0] == 2:
		subsection = {k:v for (k,v) in data.items() if k >= guess[1]}
		subsection = {k:v for (k,v) in subsection.items() if k <= guess[2]}
		x = np.array(list(subsection.keys()))
		y = np.array(list(subsection.values()))
	elif guess[0] == 3:
		subsection = {k:v for (k,v) in data.items() if k >= guess[1]}
		x = np.array(list(subsection.keys()))
		y = np.array(list(subsection.values()))

	#print('x,y', x,y)
 
	popt, pcov = curve_fit(loglog, x, y, p0, maxfev = 5000)	
	
	return popt, x

def find_fit(mass_counts, mass_guess, p0, iterations):
	random.seed(234545)
	
	for n in range(iterations):
		fit_n = [] #current fit
		sd_n = [] #current sd

		best_error = 10e6
		print("n:", n)
		#choose random masses in the range
		break1 = random.uniform(mass_guess[0][0], mass_guess[0][1])
		break2 = random.uniform(mass_guess[1][0], mass_guess[1][1])
		break3 = random.uniform(mass_guess[2][0], mass_guess[2][1])
		#goes thru each segment 
		try:
			for k in range(4):
				if k == 0:
					breaks = [0, break1]
				elif k == 1:
					breaks = [1, break1, break2]
				elif k == 2:
					breaks = [2, break2, break3]
				elif k == 3:
					breaks = [3, break3]
				
				try:
					popt, mass_arr = calc_fit(mass_counts, breaks, p0[k])
					#print('mass_arr',mass_arr)
					
					print('popt', popt)
					count_fit = [loglog(i,*popt) for i in mass_arr]
					#print('count fit', count_fit)
					fit_n.append([mass_arr, count_fit])		

				except ValueError:
					print('no vals in ydata, setting fit and sd to 0')
					fit_n.append([[0], [0]])
				except TypeError:
					print('not enough points to do least squares fit')
					fit_n.append([[0],[0]])
			#print(fit_n)

			#flatten out fit_n so that [mass_arr, count_fit]
			fit_n = np.hstack(fit_n)
			print('fit n ', fit_n)

			#calculate difference between actual count vs fit count 
			count_diff = []
			for k in range(len(fit_n)):
				print('len fit_n[k]', len(fit_n[k]))
				print('len mass_counts', len(mass_counts))
				for i in range(len(fit_n[k])-1):
					count_n = list(mass_counts.values())[i]
					if count_n == 0: #avoid division by zero
						continue
					else:
						dif_n = abs(count_n - fit_n[k][i])/count_n
					count_diff.append(dif_n) 

			avg_error = np.mean(count_diff)

			#check for better fit
			if abs(avg_error) < abs(best_error):  
				best_error = avg_error
				fit = fit_n
		except RuntimeError:
			print('RuntimeError, unable to fit')
			continue	

	fitted_counts = {}
	for i in range(len(fit[0])):
		fitted_counts[fit[0][i]] = fit[1][i]				
	return fitted_counts