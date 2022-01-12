import os 
import yt 
import pyfof
import random
import pynbody
import numpy as np 
import pandas as pd
import matplotlib.cm as cm
import pynbody.plot.sph as sph
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import track_halo_number
from statistics import mode
from scipy.stats import wasserstein_distance as wd

def get_fn(model):
	if(model == 'GM3SI1'):
		fn = '/scratch/06040/adcruz/GMs/GM4SIDM1SFBH/GM4SI1SFBH.1536.'
	elif(model == 'GM3'): 
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM4.1536gst1bwK1BH/pioneer50h243GM4.1536gst1bwK1BH.'	
	elif(model == 'GM3noBHs'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM4.1536gst1bwK1/pioneer50h243GM4.1536gst1bwK1.'
	elif(model == 'GM2'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM7.1536gst1bwK1BH/pioneer50h243GM7.1536gst1bwK1BH.'
	elif(model == 'GM2SI1'):
		fn = '/scratch/06040/adcruz/GMs/si1pioneer50h243GM7.1536gst1bwK1BH/si1pioneer50h243GM7.1536gst1bwK1BH.'
	elif(model == 'GM2noBHs'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM7.1536gst1bwK1/pioneer50h243GM7.1536gst1bwK1.'
	elif(model == 'P0'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243.1536gst1bwK1BH/pioneer50h243.1536gst1bwK1BH.'
	elif(model == 'P0noBHs'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243.1536gst1bwK1/pioneer50h243.1536gst1bwK1.'
	elif(model == 'GM1'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM1.1536gst1bwK1BH/pioneer50h243GM1.1536gst1bwK1BH.'
	elif(model == 'GM1noBHs'):
		fn = '/scratch/06040/adcruz/GMs/pioneer50h243GM1.1536gst1bwK1/pioneer50h243GM1.1536gst1bwK1.'
	else:
		print('model isnt currently an option. Add it or try again.')
	return fn

def timetrace_clumps(model):
	pyfof_HI_clumps.get_fn(model)
	data = pd.read_table(fn +'timetrace_grps.dat', sep='\s+')
	outputs = np.flip(np.unique(data['output1']))
	
	outfn = 'GMs_galaxy_properties_3456.dat'
	outfile = open(outfn, 'w')
	outfile.write('model output rvir[kpc] smass[Msol] dmass[Msol] gmass[Msol]\n')
	
	for i in range(len(outputs)-1): 
		data_output1 = data[data['output1'] == outputs[i]]
		data_output2 = data[data['output1'] == outputs[i+1]]
		grpmatch = data_output1['grpmatch']
		for j in range(len(grpmatch)):
			if(grpmatch[j] < 0):
				print('this clump no longer exists')
		data_output1_tracegrps = data_output1[data_output1['grpmatch']>0]
		data_output1_tracegrps = data_output1_tracegrps[data_output1_tracegrps['fraction'] >=0.5]
		uni_tracegrps = np.unique(data_output1_tracegrps['grpmatch'])
		tracegrp_map = np.in1d(data_output2['grp'], uni_tracegrps)

def get_haloproperties(models): 
	tsdata    = pd.read_table('../datfiles/GMs_track_halos_v2.dat', sep='\s+')
	tsgm3data = pd.read_table('../datfiles/GM3_track_halos.dat', sep='\s+')
	tsdata    = pd.concat([tsdata, tsgm3data])
	outfn = 'GMs_galaxy_properties_3456.dat'
	outfile = open(outfn, 'w')
	outfile.write('model output rvir[kpc] smass[Msol] dmass[Msol] gmass[Msol]\n')
	 
	for i in range(len(models)): 
		model_data = tsdata[tsdata['model'] == models[i]]
		#outputs = np.array(model_data['output'])
		#halos   = np.array(model_data['halo'])
		outputs = ['003456']
		halos   = [1]
		for j in range(len(outputs)): 
			#try:
			h1   = get_mainHalo(models[i], outputs[j], halos[j])
			rvir = pynbody.analysis.halo.virial_radius(h1, overden=200)
			halo = h1[h1['r'] < rvir]
			smass = sum(halo.s['mass'])
			dmass = sum(halo.d['mass'])
			gmass = sum(halo.g['mass'])
			outfile.write('%s %s %.2f %.2f %.2f %.2f\n' % (models[i], outputs[j], rvir, smass, dmass, gmass))
			print('done with model ' + models[i] + ' output ' + outputs[j])
			#except: 
			#	print('something is up with model ' + models[i] + ' output ' + outputs[j])
	outfile.close()

def HI_covering_frac(models, output): 
	outfn = 'GMs_covering_frac_' + output +  '.dat'
	outfile = open(outfn, 'w')
	outfile.write('model output z covering_frac\n')
	
	for i in range(len(models)):
		fn = get_fn(models[i])
		s = pynbody.load(fn + output)
		s.physical_units()
		z = s.properties['z']
		h = s.halos()
		h1 = h[1]
		pynbody.analysis.halo.center(h1, mode='com')
		h1 = cut_gal(h1.g)
		add_NHI(h1)
		h1_total = h[1]
		rvir = pynbody.analysis.halo.virial_radius(h1_total)
		image2 = sph.image(h1, qty="HIn",width=2*rvir,cmap='cividis', resolution=1000, av_z ='rho')
		HIflatten = np.ndarray.flatten(image2)
		x = np.linspace(-rvir, rvir, 1000)
		X, Y = np.meshgrid(x, x)
		R = (X*X + Y*Y)**(1/2)
		Rmap = R < float(rvir)
		image_rvir = image2[Rmap]
		image_rvir_flatten = np.ndarray.flatten(image_rvir)
		imagemap = image_rvir_flatten > 4e-8
		cover_frac = (len(image_rvir_flatten[imagemap])/len(image_rvir_flatten) )*100
		outfile.write('%s %s %.2f %.2f\n' % (models[i], output, z, cover_frac))
		print('finishing model... ' + models[i])
	outfile.close()

def timetrace_grpclumps(model, link_len, HI_cut, min_m): 
	fn2 = get_fn(model)
	outfn = fn2 + 'timetrace_grps_lowtohigh' + '.dat'
	
	#check if outfile already exists	
	if os.path.exists(outfn):
		outfile = open(outfn, 'r')
		lines = [line for line in outfile]
		outfile.close()
		
		#finds what the last group in the file is 	
		if len(lines) == 1 :
			#file only has column names wriitten
			last_timestep_in_file = 0
		else:
			last_timestep_in_file = len(lines) - 1
	#creates an outfile if one doesnt already exist
	else:
		last_timestep_in_file = 0
		with open(outfn, 'a') as outfile:
			outfile.write('model output1 output2 grp grpmatch fraction\n')
	
	tsdata    = pd.read_table('../datfiles/GMs_track_halos_v2.dat', sep='\s+')
	tsgm3data = pd.read_table('../datfiles/GM3_track_halos.dat', sep='\s+')
	tsdata    = pd.concat([tsdata, tsgm3data])
	
	model_data = tsdata[tsdata['model'] == model]
	#go from low to high instead
	output = np.flip(np.array(model_data['output']))
	halo   = np.flip(np.array(model_data['halo']))
	
	for j in range(last_timestep_in_file, len(output)):
		try: 
			if(len(str(output[j+1])) == 4):
				tsoutfn = fn2 + '00' + str(output[j+1])
				s = pynbody.load(tsoutfn)
				clump_grp = np.loadtxt(tsoutfn + '.clump.grp')
				pyfof_grps, grp_iords = clumps_and_iords(model, '00' + str(output[j]), link_len, HI_cut, min_m, halo[j])
			else: 
				tsoutfn = fn2 + '000' + str(output[j+1])
				s = pynbody.load(tsoutfn)
				clump_grp = np.loadtxt(tsoutfn + '.clump.grp')
				pyfof_grps, grp_iords = clumps_and_iords(model, '000' + str(output[j]), link_len, HI_cut, min_m, halo[j])
			
			s.physical_units()
			s.g['clump.grp'] = clump_grp
			for i in range(len(grp_iords)):
				grpmap = np.in1d(s.g['iord'], grp_iords[i])
				unigrp = np.unique(s.g[grpmap]['clump.grp'])
				print("partices in adj ts grps " + str(s.g[grpmap]['clump.grp']))
				print("unique grps for index+1 = " + str(i+1) + ' is ' + str(unigrp))	
				try:
					matching_grp = mode(s.g[grpmap]['clump.grp'])
					grp = i+1
				except: 
					if(len(unigrp)==len(s.g[grpmap])):
						matching_grp = min(unigrp)
						grp = i+1
					elif(len(unigrp)==2):
						matching_grp = min(unigrp)
						grp = i+1
				fraction = len(s.g[grpmap]['clump.grp'][s.g[grpmap]['clump.grp'] == matching_grp])/len(s.g[grpmap])
				with open(outfn, 'a') as outfile: 
					outfile.write('%s %s %s %i %i %.2f\n' % (model, output[j], output[j+1], grp, matching_grp, fraction))	
		except: 
			print(tsoutfn + 'or its clump.grp file are missing.')	

def clumps_and_iords(model, output, link_len, HI_cut, min_m, halo):
	fn = get_fn(model)
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	add_NHI(s2)
	if (model == 'GM3noBHs'):
		h2 = s2.halos(write_fpos = False)
	else:
		h2 = s2.halos()
	h21 = h2[halo]
	pynbody.analysis.halo.center(h21, mode = 'com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	
	s2 = cut_gal(s2)
	z = s2.properties['z']

	h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]
	
	pyn_data =  np.array(h21_HIcut['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)
	pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
	pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)
	return pyfof_grp_minm, pyfof_iord 

def clump_grp(model, output, link_len, HI_cut, min_m, halo):
	outfile = get_fn(model) + output + '.clump.grp'	
	if(os.path.isfile(outfile)): 
		print("file " + outfile + " already exists.")
	else:
		fn = get_fn(model)
		s = pynbody.load(fn + output)
		s.physical_units()
		add_NHI(s)
		if (model == 'GM3noBHs'):
			h2 = s.halos(write_fpos = False)
		else:
			h2 = s.halos()
		h21 = h2[halo]
		pynbody.analysis.halo.center(h21, mode = 'com')
		h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	
		s2 = cut_gal(s)
		z = s2.properties['z']

		h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]
	
		pyn_data =  np.array(h21_HIcut['pos'])
		pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)

		pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
		pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)

		for i in range(len(pyfof_iord)): 
			iord_map = np.in1d(s.g['iord'], pyfof_iord[i])
			s.g[iord_map]['clump_grp'] = i+1*np.ones(len(s.g[iord_map]))
			#print("s.g[iord_map]['clump_grp'] = ", s.g[iord_map]['clump_grp'])
		pyfof_iord = np.array(pyfof_iord)
		iords = np.hstack(pyfof_iord) 
		iord_map = np.in1d(s.g['iord'], iords) 
		#print('all iord map = ', iord_map)
		#iords not in clumps 
		notclump_iord = np.logical_not(iord_map)
		#print('iords not in clumps = ', notclump_iord)
		s.g[notclump_iord]['clump_grp'] = -1*np.ones(len(s.g[notclump_iord]))
		#print("s.g['clump_grp'] = ", s.g['clump_grp'])
		print('saving file.. ' + outfile)
		np.savetxt(get_fn(model) + output + '.clump.grp', s.g['clump_grp']) 	
	
def N_pyfof_clumps(model, output, link_len, HI_cut, min_m, halo): 
	fn = get_fn(model)
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	add_NHI(s2)
	if (model == 'GM3noBHs'):
		h2 = s2.halos(write_fpos = False)
	else:
		h2 = s2.halos()
	h21 = h2[halo]
	pynbody.analysis.halo.center(h21, mode = 'com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	
	s2 = cut_gal(s2)
	z = s2.properties['z']

	h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]
	
	pyn_data =  np.array(h21_HIcut['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)

	pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
	pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)
	pyfof_iord = np.array(pyfof_iord)
	#print('pyfof_iord = ' + str(pyfof_iord))
	iords = np.hstack(pyfof_iord) 
	#print('flattened iords = ' + str(iords))
	iord_mask = np.in1d(s2.g['iord'], iords)
	print('number of iords = ' + str(len(iords)))
	print('unique n of iords = ' + str(len(np.unique(iords))))
	cold_cgm_gas   = cold_gas(h21_HIcut)
	cold_clump_gas = cold_gas(s2.g[iord_mask])
	
	return z, len(pyfof_grp_minm), sum(cold_cgm_gas['mass']), sum(cold_clump_gas['mass'])

def pyfof_clump_data(model, output, link_len, HI_cut, min_m):
	print('model: ' + model)
	print('output: ' + output)
	
	fn = get_fn(model)

	outfn = '/scratch/08263/tg875625/scripts/datfiles/' + model + '_' + output + '_HI' + str(HI_cut) + '_' + str(link_len) + 'kpc_clump_data.dat'
		
	#check if outfile already exists	
	if os.path.exists(outfn):
		outfile = open(outfn, 'r')
		lines = [line for line in outfile]
		outfile.close()
		
		#finds what the last group in the file is 	
		if len(lines) == 1 :
			#file only has column names wriitten
			last_group_in_file = 0
		else:
			last_group_in_file = len(lines)
	#creates an outfile if one doesnt already exist
	else:
		last_group_in_file = 0
		with open(outfn, 'a') as outfile:
		
			outfile.write('model linking_length pyfof_grp N clump_mass[Msol] com_x[kpc] com_y[kpc] com_z[kpc] max_r[kpc] clump_r_avg[kpc] r[kpc] avg_temp[K] avg_pressure[Pa] avg_nHden[cm^-3] vdist[km/s] entropy[K/(Msolkpc**3)**(2/3)]\n')

	s2 = pynbody.load(fn + output)
	s2.physical_units()
	add_NHI(s2)
	if (model == 'GM3noBHs'):
		h2 = s2.halos(write_fpos = False)
	else:
		h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode = 'com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	s2 = cut_gal(s2)

	z = s2.properties['z']

	h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]
	
	pyn_data =  np.array(h21_HIcut['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)

	pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
	pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)
		
	s_com  = pynbody.analysis.halo.center_of_mass(s2)

	for i in range(last_group_in_file, len(pyfof_grp_minm)):
		print('group ' + str(i) + ' of ' + str(len(pyfof_grp_minm)-1)
)
		#comi = group i's COM
		comi = get_pyfof_com_1(s2.g, pyfof_iord[i]).in_units('kpc')
		print('comi = ' + str(comi))

		iord_map = np.in1d(s2.g['iord'], pyfof_iord[i])
		pyfof_grpi = s2.g[iord_map]
		#print('pos of pyfof_grpi before = ' + str(pyfof_grpi['pos']))	

		#grp_com_translate(s2.g, comi)
		pyfof_grpr = max(pyfof_grpi['r'].in_units('kpc'))
		print('pyfof_grpr = ' + str(pyfof_grpr))
		clumpgas = s2.g[s2.g['r'].in_units('kpc') <= pyfof_grpr]
		#grp_com_translate(s2.g, -comi)
		h1_com = pynbody.analysis.halo.center_of_mass(h21).in_units('kpc')
		
		pyfof_pos    = pyfof_grpi['pos'].in_units('kpc')
		pyfof_vel    = pyfof_grpi['vel'].in_units('km s**-1')
		print('pyfof_vel = ' + str(pyfof_vel))
		pyfof_massi  = sum(pyfof_grpi['mass'].in_units('Msol'))
		pyfof_velcom = pynbody.analysis.halo.center_of_mass_velocity(pyfof_grpi).in_units('km s**-1')
	
		#avg quanitites 
		avg_temp = pyfof_grpi['temp'].mean()
		avg_pres = pyfof_grpi['p'].mean()
		
		#density 
		mp = pynbody.array.SimArray(1.67262e-27)
		mp.units = 'kg'
		
		grp_rho = np.array(pyfof_grpi['rho'].in_units('kg cm**-3'))
		grp_n = (pyfof_grpi['HI'] * grp_rho / mp)
		avg_n = grp_n.mean()
		print('avg_n = ' + str(avg_n))		

		#entropy 
		pyfof_entropy = pyfof_grpi['temp'] / (grp_rho)**(2/3)
		avg_entropy = pyfof_entropy.mean()
		print('avg_entropy = ' + str(avg_entropy))	
	
		#pyfof com distance from center of CGM 
		pyfof_cgm_com = h1_com - comi 
		cgm_com_dist   = ((pyfof_cgm_com[0])**2 + (pyfof_cgm_com[1])**2 + (pyfof_cgm_com[2])**2)**(0.5)  
		
		# average radial distance of clump members to com of clump 
		pyfof_rmr0 = pyfof_pos - comi 
		rmr0x      = pyfof_rmr0[:, 0] 
		rmr0y      = pyfof_rmr0[:, 1] 
		rmr0z      = pyfof_rmr0[:, 2]
		rad_dist   = (rmr0x**2 + rmr0y**2 + rmr0z**2)**(0.5)
		r_dist_avg  = sum(rad_dist) / len(rad_dist) 
		maxr       = max(rad_dist)
		print('r_dist_avg in kpc? = ' + str(r_dist_avg))
		#pyfof_comkpc = comi.in_units('kpc')

		# rms velocity for each clump 
		pyfofvmv0 = (pyfof_vel - pyfof_velcom)
		print('pyfofvmv0 = ' + str(pyfofvmv0))		
		vmv0x   = pyfofvmv0[:, 0] 
		vmv0y   = pyfofvmv0[:, 1] 
		vmv0z   = pyfofvmv0[:, 2]
		rad_vel = (vmv0x**2 + vmv0y**2 + vmv0z**2)**(0.5)
		avg_vel = rad_vel.mean() 
	
		with open(outfn, 'a') as outfile: 
			outfile.write('%s %.2f %i %i %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.10f %.2f %.2f\n' %(model, 
			link_len, i, len(pyfof_grpi), 
			pyfof_massi, comi[0], comi[1], 
			comi[2], maxr, r_dist_avg, cgm_com_dist, 
			avg_temp, avg_pres, avg_n, avg_vel, avg_entropy))
	
def velHist_pyfofclumps(model, output, link_len, HI_cut, min_m, z1,datfile): 
	
	fn = get_fn(model)

	s2 = pynbody.load(fn + output)
	s2.physical_units()
	add_NHI(s2)
	h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode='com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	s2 = cut_gal(s2)

	h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]
	
	pyn_data =  np.array(h21_HIcut['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)

	pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
	pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)
	
	 
	fig = plt.figure(figsize = (15, 15))
	bins = np.logspace(np.log10(2e-1), np.log10(9e1), 20)
	
	int_rWdist = []
	hot_rWdist = []
	int_vrWdist = []
	hot_vrWdist = []
	int_hot_vrWdist = []
	int_hot_rWdist = []
	
	cold_inter_clump_dict = open("cold_inter_" + datfile, 'w')
	cold_inter_clump_dict.write('inx maxr cold_mass inter_mass Wvx Wvy Wvz Wvr Wvt Wx Wy Wz Wr\n')
	
	cold_hot_clump_dict = open("cold_hot_" + datfile, 'w')
	cold_hot_clump_dict.write('inx maxr cold_mass hot_mass Wvx Wvy Wvz Wvr Wvt Wx Wy Wz Wr\n')
	
	inter_hot_clump_dict = open("inter_hot_" + datfile, 'w')
	inter_hot_clump_dict.write('inx maxr inter_mass hot_mass Wvx Wvy Wvz Wvr Wvt Wx Wy Wz Wr\n')
	
	maxr_int  = []
	maxr_hot  = []
	for i in range(len(pyfof_grp_minm)):
		r = random.random()
		g = random.random()
		b = random.random()
		comi = get_pyfof_com_1(s2.g, pyfof_iord[i])
		print('comi = ' + str(comi))
		iord_map = np.in1d(s2.g['iord'], pyfof_iord[i])
		pyfof_grpi = s2.g[iord_map]
		print('pos of pyfof_grpi before = ' + str(pyfof_grpi['pos']))	
		grp_com_translate(s2.g, comi)
		pyfof_grpr = max(pyfof_grpi['r'].in_units('kpc'))
		print(pyfof_grpr)
		clumpgas = s2.g[s2.g['r'].in_units('kpc') <= pyfof_grpr]
		coldgas = cold_gas(clumpgas)
		hotgas  = hot_gas(clumpgas)
		intergas  = inter_gas(clumpgas)
		grp_com_translate(s2.g, -comi)
		h1_com = pynbody.analysis.halo.center_of_mass(h21)
		print('h1_com com translate back ' + str(h1_com))
		
		if(len(intergas) > 0): 
			#plt.hist(coldgas['vr'], bins = bins, color = [r, b, g], label = model + r'T < 10$^{5}$ K')
			#plt.hist(intgas['vr'], bins = bins, color = [r, b, g],alpha=0.25, label = model + r'10$^{5}$ < T < 10$^{6}$ K')
			maxr_int.append(pyfof_grpr)
			int_vrWdist.append(wd(coldgas['vr'], intergas['vr']))
			int_rWdist.append(wd(coldgas['r'], intergas['r']))
			cold_inter_clump_dict.write('%i %f %f %f %f %f %f %f %f %f %f %f %f \n'%(i, pyfof_grpr, sum(coldgas['mass']), sum(intergas['mass']), wd(coldgas['vel'][:,0], intergas['vel'][:,0]), wd(coldgas['vel'][:,1], intergas['vel'][:,1]), wd(coldgas['vel'][:,2], intergas['vel'][:, 2]), wd(coldgas['vr'], intergas['vr']), wd(coldgas['vt'], intergas['vt']), wd(coldgas['pos'][:,0], intergas['pos'][:, 0]), wd(coldgas['pos'][:, 1], intergas['pos'][:, 1]), wd(coldgas['pos'][:,2], intergas['pos'][:, 2]), wd(coldgas['r'], intergas['r'])))
			print('calculated wasser dist for cold and inter gas') 
		else: 
			print('no intermed gas for grp ' + str(i))

		if(len(hotgas) > 0):
			#plt.hist(hotgas['vr'], bins = bins, color = [r, b, g],edgecolor='k', alpha=0.25, label = model + r' T > 10$^{6}$ K')
			maxr_hot.append(pyfof_grpr)
			hot_vrWdist.append(wd(coldgas['vr'], hotgas['vr']))
			hot_rWdist.append(wd(coldgas['r'], hotgas['r']))
			cold_hot_clump_dict.write('%i %f %f %f %f %f %f %f %f %f %f %f %f \n'%(i, pyfof_grpr, sum(coldgas['mass']), sum(hotgas['mass']), wd(coldgas['vel'][:,0], hotgas['vel'][:,0]), wd(coldgas['vel'][:,1], hotgas['vel'][:,1]), wd(coldgas['vel'][:,2], hotgas['vel'][:,2]), wd(coldgas['vr'], hotgas['vr']), wd(coldgas['vt'], hotgas['vt']), wd(coldgas['pos'][:,0], hotgas['pos'][:,0]), wd(coldgas['pos'][:,1], hotgas['pos'][:,1]), wd(coldgas['pos'][:,2], hotgas['pos'][:,2]), wd(coldgas['r'], hotgas['r'])))

			if(len(intergas) > 0):
				int_hot_vrWdist.append(wd(hotgas['vr'], intergas['vr']))
				int_hot_rWdist.append(wd(hotgas['r'], intergas['r']))
				inter_hot_clump_dict.write('%i %f %f %f %f %f %f %f %f %f %f %f %f \n'%(i, pyfof_grpr, sum(intergas['mass']), sum(hotgas['mass']), wd(intergas['vel'][:,0], hotgas['vel'][:,0]), wd(intergas['vel'][:,1], hotgas['vel'][:,1]), wd(intergas['vel'][:,2], hotgas['vel'][:,2]), wd(intergas['vr'], hotgas['vr']), wd(intergas['vt'], hotgas['vt']), wd(intergas['pos'][:,0], hotgas['pos'][:,0]), wd(intergas['pos'][:,1], hotgas['pos'][:,1]), wd(intergas['pos'][:,2], hotgas['pos'][:,2]), wd(intergas['r'], hotgas['r'])))
				
			print('calculated wasser dist for cold and hot gas') 
		#except:
		else:				
			print('no hot gas for grp ' + str(i))
	
	#plt.loglog()	
	#plt.xlabel('grp number', fontsize = 20)
	#plt.ylabel(r'v$_{r}$ [km/s]', fontsize = 20)
	#plt.tick_params(labelsize = 16)
	#plt.legend(fontsize = 14)
	#outfile = model + output + '_grpvsvr_cold.pdf'
	#plt.savefig(outfile)
	#plt.show()
	
	fig = plt.figure(figsize = (15, 15))
	plt.scatter(int_rWdist, int_vrWdist, color = 'green', s =200, label = model + r'10$^{5}$ < T < 10$^{6}$ K ') 
	plt.scatter(hot_rWdist, hot_vrWdist, color = 'red', s=200, label = model + r' T > 10$^{6}$ K')
	plt.scatter(int_hot_rWdist, int_hot_vrWdist, color = 'blue',s=200, label = model + 'hot/inter')
	plt.loglog()	
	plt.xlabel('clump radius [kpc]', fontsize = 20)
	plt.ylabel(r'D$_{was}$ v$_{r}$', fontsize = 20)
	plt.tick_params(labelsize = 20)
	plt.legend(fontsize = 18)
	outfile2 = model + output + '_clump_vs_Wasdist.pdf'
	plt.savefig(outfile2)
	plt.show()
	
	cold_inter_clump_dict.close()
	cold_hot_clump_dict.close()
	inter_hot_clump_dict.close()

def pyfof_clumps(model, output, link_len, HI_cut, plot, plt_all, HIarray, show, rotate, min_m, res, width, vmin, vmax, z1):
	"""
	rotate = array of x, y, z rotation angles
	"""
	fn = get_fn(model)
	
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode='com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	s2 = cut_gal(s2)

	#add N HI 	
	mp = pynbody.array.SimArray(1.67262 * 10**(-27))
	mp.units = 'kg'
	HI_mp = s2.g['HI'] * s2.g['rho'].in_units('kg cm**-3') / ( mp ) 
	s2.g['HIn'] = HI_mp
	s2.g['HIn'].units = 'cm**-3'

	if(HIarray == 'yt'):
		ds = yt.load(fn + output)
		ad = ds.all_data()
		trident.add_ion_fields(ds, ions=['H I'], ftype="gas")
		pyn_HI = pynbody.array.SimArray(ad[('gas', 'H_p0_number_density')])
		pyn_HI.units = 'cm^-3'
		s2.g['HI'] = pyn_HI
	elif(HIarray == 'pynbody'): 
		print('pynbody HI array being used')
	else: 
		print('didnt pick aval option. Pick pynbody or yt.')	

	h21_HIcut = h21_rcut[h21_rcut['HIn']  > HI_cut]

	#rotate view
	if(np.abs(rotate[0]) > 0 ):
		s2.rotate_x(rotate[0])
	elif(np.abs(rotate[1]) > 0): 
		s2.rotate_y(rotate[1])
	elif(np.abs(rotate[2]) > 0): 
		s2.rotate_z(rotate[2])

	#pos after rotation so you dont have to keep track of iords? 	
	pyn_data =  np.array(h21_HIcut['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)
	
	if(plot): 
		pyn_colors = cm.twilight(np.linspace(0, 1, len(pyn_groups)))
		print('pyn_colors = ' + str(pyn_colors))
	
		fig = plt.figure(figsize =(15, 15))
		if(plt_all):
			image2 = sph.image(s2.g, qty="HIn",width=width,cmap='cividis', resolution=res, av_z ='rho')	
			datfile = 'all_s_' + model + '_' + output + 'rotatex_' + str(rotate[0]) + '_y_' + str(rotate[1]) + '_z_' + str(rotate[2]) + 'min_m' + str(min_m) + '_e' + str(link_len)  + '_HIcutfix_HIsph_pyfof.dat'
		else: 
			image2 = sph.image(h21_HIcut, qty="HIn",width=width,cmap='cividis', resolution=res, av_z ='rho') 
			datfile = 'h_r_HIcut_' + model + '_' + output + 'rotatex_' + str(rotate[0]) + '_y_' + str(rotate[1]) + '_z_' + str(rotate[2]) + 'min_m' + str(min_m) + '_e' + str(link_len) + '_HIcutfix_HIsph_pyfof.dat'
			
		x = np.linspace(-width/2, width/2, res)
		X, Y = np.meshgrid(x, x)
	
		#fig = plt.figure(figsize = (15, 15))
		fig, ax = plt.subplots(figsize = (15, 15))
		pim = ax.pcolormesh(X, Y, image2, cmap = 'cividis', norm = colors.LogNorm(vmin, vmax) )
		ax.annotate(model, xy = ((-width/2)+50, (width/2)-50 ), color = 'white', fontsize = 16)
		ax.annotate('z = %.2f'%z1, xy = ((width/2)- 100, (width/2) - 50), color = 'white', fontsize = 16 )
		ax.annotate('m = ' + str(min_m) + ' e = ' + str(link_len) + ' kpc', xy = ((width/2)- 125, -((width/2) - 50)), color = 'white', fontsize = 16 )
		ax.tick_params(labelsize = 16)
		cb = fig.colorbar(pim)
		cb.ax.tick_params(labelsize = 16)
		cb.ax.set_ylabel(r'n$_{HI}$ [cm$^{-3}$]', fontsize = 16)
		
		if(rotate[0] == -90):	
			plt.xlabel('x [kpc]', fontsize = 16)
			plt.ylabel('z [kpc]', fontsize = 16)
		elif(rotate[1] == 90): 
			plt.xlabel('z [kpc]', fontsize = 16)
			plt.ylabel('y [kpc]', fontsize = 16)
		else:
			plt.xlabel('x [kpc]', fontsize = 16)
			plt.ylabel('y [kpc]', fontsize = 16)
		
		pyfof_grp_minm = grp_min_m(pyn_groups, min_m)
		
		if(plt_all):
			pyfof_iord = get_iords(s2.g, h21_HIcut, pyfof_grp_minm)
		else: 
			pyfof_iord = get_iords(h21_HIcut, h21_HIcut, pyfof_grp_minm)

		
		i = 0 
		clump_dict = open(datfile, 'w')
		clump_dict.write('inx clump_mass cold_mass inter_mass hot_mass max_comdist N Ncold Ninter Nhot \n')
		for k,c in zip(pyfof_grp_minm, pyn_colors):
			r = random.random()
			g = random.random()
			b = random.random()
			ax.scatter(pyn_data[k,0], pyn_data[k,1], color=[r,b,g], s=3)
			if(plt_all):
				comi = get_pyfof_com_1(s2.g, pyfof_iord[i])
				print('comi = ' + str(comi))
				iord_map = np.in1d(s2.g['iord'], pyfof_iord[i])
				pyfof_grpi = s2.g[iord_map]
				print('pos of pyfof_grpi before = ' + str(pyfof_grpi['pos']))	
				grp_com_translate(s2.g, comi)
				pyfof_grpr = max(pyfof_grpi['r'].in_units('kpc'))
				print(pyfof_grpr)
				clumpgas = s2.g[s2.g['r'].in_units('kpc') <= pyfof_grpr]
				coldgas = cold_gas(clumpgas)
				hotgas  = hot_gas(clumpgas)
				intgas  = inter_gas(clumpgas)
				clump_dict.write('%i %f %f %f %f %f %i %i %i %i \n'%(i, sum(clumpgas['mass']), sum(coldgas['mass']), sum(intgas['mass']), sum(hotgas['mass']), pyfof_grpr, len(clumpgas), len(coldgas), len(intgas), len(hotgas)))
				grp_com_translate(s2.g, -comi)
				h1_com = pynbody.analysis.halo.center_of_mass(h21)
				print('h1_com com translate back ' + str(h1_com))
				print(h1_com)
			else: 
				comi = get_pyfof_com_1(h21_HIcut, pyfof_iord[i])
				grp_com_translate(h21_HIcut, comi)
				iord_map = np.in1d(h21_HIcut['iord'], pyfof_iord[i])
				pyfof_grpi = h21_HIcut[iord_map]
				pyfof_grpr = max(pyfof_grpi['r'])
				grp_com_translate(h21_HIcut, -comi)
			print(pyfof_grpr)
			circle = plt.Circle(comi, pyfof_grpr, color=[r,b,g], alpha=0.3)				
			ax.add_patch(circle)
			i += 1
		clump_dict.close()

		ax.set_xlim(-width/2, width/2)
		ax.set_ylim(-width/2, width/2)

		if(plt_all): 
			outfile = 'all_s_' + model + '_' + output + 'rotatex_' + str(rotate[0]) + '_y_' + str(rotate[1]) + '_z_' + str(rotate[2]) + 'min_m' + str(min_m) + '_e' + str(link_len)  + '_HIcutfix_HIsph_pyfof.pdf'
		else: 
			outfile = 'h_r_HIcut_' + model + '_' + output + 'rotatex_' + str(rotate[0]) + '_y_' + str(rotate[1]) + '_z_' + str(rotate[2]) + 'min_m' + str(min_m) + '_e' + str(link_len) + '_HIcutfix_HIsph_pyfof.pdf'
			
		plt.savefig(outfile)
	
		if(show == True):
			plt.show() 
		else: 
			plt.clf()		
	else: 
		return h21_HIcut, pyn_groups, pyn_data

def pyfof_clump_mass(model, output, link_len, HI_cut, HIarray): 
	fn = get_fn(model)
	
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode='com')
	h21_rcut = h21.g[h21.g['r'].in_units('kpc') > 15]
	
	ds = yt.load(fn + output)
	ad = ds.all_data()

	if(HIarray == 'yt'):
		trident.add_ion_fields(ds, ions=['H I'], ftype="gas")
		pyn_HI = pynbody.array.SimArray(ad[('gas', 'H_p0_number_density')])
		pyn_HI.units = 'cm^-3'
		s2.g['HI'] = pyn_HI
	elif(HIarray == 'pynbody'): 
		print('pynbody HI array being used')
	else: 
		print('didnt pick aval option. Pick pynbody or yt.')	
	

	h21_HI = h21_rcut[h21_rcut['HI'] > HI_cut]

	pyn_data =  np.array(h21_HI['pos'])
	pyn_groups = pyfof.friends_of_friends(pyn_data, link_len)

	clump_mass = [] 
	
	for i in range(len(pyn_groups)):
		clump_mass.append(sum(h21_HI['mass'][pyn_groups[i]].in_units('Msol')))

	outfile = model + '_' + output + '_HIcut_' + str(HI_cut) + '_'+ 'link_len_' + str(link_len)  + '_kpc.txt' 
	np.savetxt(outfile, clump_mass)

def hot_gas(pyn_halo):
	return pyn_halo.g[pyn_halo.g['temp'].in_units('K') > 10**6]

def cold_gas(pyn_halo):
	return pyn_halo.g[pyn_halo.g['temp'].in_units('K') < 10**5]

def inter_gas(pyn_halo):
	inter_gas1 = pyn_halo.g[pyn_halo.g['temp'] > 10**5]
	return inter_gas1[inter_gas1['temp'] < 10**6]

def halo_no_sats(halo, n_subs):
	halo1 = halo[halo['amiga.grp'] != 1+n_subs]
	return halo1[halo1['amiga.grp'] != 2+n_subs]

def cut_gal(halo): 
	return halo.g[halo.g['r'].in_units('kpc') > 15]

def grp_min_m(pyfof_grps, m): 
	pyfof_grps_m = []
	for i in range(len(pyfof_grps)):
    		if(len(pyfof_grps[i]) > (m-1) ):
        		pyfof_grps_m.append(pyfof_grps[i])
	return pyfof_grps_m

def add_NHI(subsnap):
	#add N HI 	
	mp = pynbody.array.SimArray(1.67262 * 10**(-27))
	mp.units = 'kg'
	HI_mp = subsnap.g['HI'] * subsnap.g['rho'].in_units('kg cm**-3') / ( mp ) 
	subsnap.g['HIn'] = HI_mp
	subsnap.g['HIn'].units = 'cm**-3'

def grp_com_translate(h, com):
	pynbody.transformation.translate(h, -com)

def get_iords(subsnap, h1_HIcut, pyfof_grp_minm): 
	pyfof_iord = []
	for i in range(len(pyfof_grp_minm)):
		iord_map = np.in1d(subsnap['iord'], h1_HIcut[pyfof_grp_minm[i]]['iord'])
		pyfof_iord.append(np.array(subsnap[iord_map]['iord']))
		#pyfof_iord = np.array(pyfof_iord)
	return pyfof_iord

def get_pyfof_com(h1_r, pyfof_grp_minm, pyfof_iord): 
	pyfof_com = []  
	for i in range(len(pyfof_grp_minm)):
    		iord_map = np.in1d(h1_r['iord'], pyfof_iord[i])
    		pyfof_grpi = h1_r[iord_map]
    		pyfof_com.append(pynbody.analysis.halo.center_of_mass(pyfof_grpi))
	return pyfof_com

def get_pyfof_com_1(h1_r, pyfof_iord_i):
	iord_map = np.in1d(h1_r['iord'], pyfof_iord_i)
	pyfof_grpi = h1_r[iord_map]
	pyfof_comi = pynbody.analysis.halo.center_of_mass(pyfof_grpi)
	return pyfof_comi 

def gasClump_temp_frac(model, output, z, m): 
	h1_HIcut, pyfof_grps, pyfof_pos = pyfof_clumps(model, output, 1, 4e-8, False, False, 'pynbody', False, [0,0, 0], 2, 1000, 600, 4e-8, 5e-2, z)
	
	h1 = get_mainHalo(model, output)
	h1_r = cut_gal(h1)
	
	pyfof_grp_m = grp_min_m(pyfof_grps, m)

	pyfof_iord = get_iords(h1_HIcut, h1_HIcut, pyfof_grp_m)

	pyfof_com  = get_pyfof_com(h1_r, pyfof_grp_m, pyfof_iord)
	
	clump_mass = []
	cold_mass  = [] 
	hot_mass   = [] 
	inter_mass = [] 
	max_radius = [] 

	for i in range(len(pyfof_grp_m)):
		#grp_com_translate(h1_r, pynbody.analysis.halo.center_of_mass(h1))
		comi = get_pyfof_com_1(h1_r, pyfof_iord[i])
		grp_com_translate(h1_r, comi)
		iord_map = np.in1d(h1_r['iord'], pyfof_iord[i])
		pyfof_grpi = h1_r[iord_map]
		clumpgas = h1_r.g[h1_r.g['r'].in_units('kpc') < max(pyfof_grpi['r'])]
		coldgas = cold_gas(clumpgas)
		hotgas  = hot_gas(clumpgas)
		intgas  = inter_gas(clumpgas)
		cold_mass.append(sum(coldgas['mass']))
		inter_mass.append(sum(intgas['mass']))
		hot_mass.append(sum(hotgas['mass']))
		clump_mass.append(sum(clumpgas['mass']))
		max_radius.append(max(pyfof_grpi['r']))
		grp_com_translate(h1_r, -comi)
		
	return cold_mass, inter_mass, hot_mass, clump_mass, pyfof_com, max_radius


def pyfof_temp_maps(model, tempmodel, output, width, res, z1):
	fn = get_fn(model)
	
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode='com')

	#cut subs 
	s_nosubs = halo_no_sats(s2, 2) 
	
	#cut gal 
	s_nosubs_r = cut_gal(s_nosubs)
	 
	#temp min and max
	temp_gal = get_mainHalo(tempmodel, output)  
	temp_gal_r = cut_gal(temp_gal) 
	min_temp = min(temp_gal_r['temp'])
	max_temp = max(temp_gal_r['temp'])
	
	#temp gas cuts 
	s_nosubs_r_cold  = cold_gas(s_nosubs_r) 
	s_nosubs_r_hot   = hot_gas(s_nosubs_r)
	s_nosubs_r_inter = inter_gas(s_nosubs_r) 

	h1_HIcut, pyfof_grps, pyfof_pos = pyfof_clumps(model, output, 1, 4e-8, False, False, 'pynbody', True, [0,0, 0], 2, res, width, 4e-8, 5e-2, z1)	

	h1_cen = pynbody.analysis.halo.center_of_mass(h21)
	h1_rvir = pynbody.analysis.halo.virial_radius(h2[1])
	
	pyfof_grps_m2 = grp_min_m(pyfof_grps, 2) 
		
	x = np.linspace(-width/2, width/2, res)
	X, Y = np.meshgrid(x, x)
	
	s_temps = [s_nosubs_r_cold, s_nosubs_r_inter, s_nosubs_r_hot]
	label = [r'T < 10$^{5}$ [K]', r'10$^{5}$ < T < 10$^{6}$ [K]', r' T > 10$^{6}$ [K]']
	images = []
	for i in range(len(s_temps)):
		image2 = sph.image(s_temps[i], qty = 'temp', resolution = res, width = width, vmin = min_temp, vmax = max_temp, cmap = 'RdBu_r')	
		images.append(image2)
		plt.clf()

	fig, ax = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (21, 7))
	label_colors = ['white', 'k', 'white']
	for i in range(len(images)): 
		pim = ax[i].pcolormesh(X, Y, images[i], cmap = 'RdBu_r', norm = colors.LogNorm(min_temp, max_temp) )
		circle = plt.Circle(h1_cen, h1_rvir, color = label_colors[i], fill = False)
		ax[i].add_patch(circle)
		ax[i].tick_params(labelsize = 14)
		ax[i].set_xlabel('x [kpc]', fontsize = 16)
		ax[i].annotate(label[i], xy = ((-width/2)+50, -(width/2)+25), color = label_colors[i], fontsize = 16)
	ax[0].set_ylabel('y [kpc]', fontsize = 16)
	cb = fig.colorbar(pim)
	cb.ax.tick_params(labelsize = 16)
	cb.ax.set_ylabel(r'Temp [K]', fontsize = 16)
	plt.annotate(model, xy = ((-width/2)+50, (width/2)-50 ), color = 'white', fontsize = 16)
	#plt.annotate('z = %.2f'%z1, xy = ((width/2)- 100, (width/2) - 50), color = 'white', fontsize = 16 )
	outfile = model + output + '_tempmap_subplots.pdf'
	plt.savefig(outfile) 
	plt.show()

	#return h1_HIcut[pyfof_grps_m2]['iord']

def pyfof_temp_hist(model, output, width, res, z1): 
	fn = get_fn(model)
	
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	h2 = s2.halos()
	h21 = h2[1]
	pynbody.analysis.halo.center(h21, mode='com')
	
	#cut gal 
	s_nosubs_r = cut_gal(h21.g)
	 
	#temp min and max 
	min_temp = min(s_nosubs_r['temp'])
	max_temp = max(s_nosubs_r['temp'])
	
	#temp gas cuts 
	s_nosubs_r_cold  = cold_gas(s_nosubs_r) 
	s_nosubs_r_hot   = hot_gas(s_nosubs_r)
	s_nosubs_r_inter = inter_gas(s_nosubs_r) 	

	fig = plt.figure(figsize = (15, 15))
	bins = np.logspace(np.log10(8e3), np.log10(3e6), 40)
	plt.hist(s_nosubs_r_cold['temp'], bins = bins, color = 'blue', alpha = 0.25, label = r'T < 10$^{5}$ K') 
	plt.hist(s_nosubs_r_inter['temp'], bins = bins, color = 'green', alpha = 0.25, label = r'10$^{5}$ K < T < 10$^{6}$ K ') 
	plt.hist(s_nosubs_r_hot['temp'], bins = bins, color = 'red', alpha = 0.25, label = r'T > 10$^{6}$ K') 
	plt.annotate(model, xy = (1e4, 5e4), fontsize = 16)	
	plt.loglog()	
	plt.xlabel('Temp [K]', fontsize = 16)
	plt.ylabel('N', fontsize = 16)
	plt.tick_params(labelsize = 16)
	plt.legend(fontsize = 14)
	outfile = model + output + '_temphist.pdf'
	plt.savefig(outfile) 
	plt.show()

def get_mainHalo(model, output, halo): 
	fn = get_fn(model)	
	s2 = pynbody.load(fn + output)
	s2.physical_units()
	h2 = s2.halos()
	h21 = h2[halo]
	pynbody.analysis.halo.center(h21, mode='com')

	return h21 	

def temp_extrema(h):
	temp = h.g['temp']
	min_temp = min(temp)
	max_temp = max(temp)
	return min_temp, max_temp 

def temp_compareHist(models, output): 

	
	fig = plt.figure(figsize = (15, 15))
	bins = np.logspace(np.log10(8e3), np.log10(3e6), 40)
	colors = ['cyan', 'purple']
	for i in range(len(models)):
		halo = get_mainHalo(models[i], output)
		temp = halo.g['temp']
		plt.hist(temp, bins = bins, color = colors[i], alpha = 0.25, label = models[i]) 
	plt.loglog()	
	plt.xlabel('Temp [K]', fontsize = 16)
	plt.ylabel('N', fontsize = 16)
	plt.tick_params(labelsize = 16)
	plt.legend(fontsize = 14)
	outfile = models[0] + output + '_compare_temphist.pdf'
	plt.savefig(outfile) 
	plt.show()




