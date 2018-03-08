#!/usr/bin/env python3
import numpy as np
import h5py
import pkg_resources
from scipy.interpolate import interp2d

class FONLL:	# dX/dpT
	def __init__(self):
		self.spectra = {}
		self.mass = {'c': 1.3, 'b':4.2}
		self.approx = {"c": lambda pT: pT/(pT**2+7.)**3.,
		 			   "b": lambda pT: pT/(pT**2+17.)**4.}
		fname = pkg_resources.resource_filename('fonll', 
				"data/Initial_Production.hdf5")
		f = h5py.File(fname, 'r')
		for nPDF in f.keys():
			ds0 = f[nPDF]
			self.spectra[nPDF] = {}
			for system in ds0.keys():
				ds1 = ds0[system]
				self.spectra[nPDF][system] = {}
				for sqrts in ds1.keys():
					ds2 = ds1[sqrts]
					self.spectra[nPDF][system][sqrts] = {}
					for specie in ds2.keys():
						ds3 = ds2[specie]
						pT	= ds3['pT'].value
						yn	= np.linspace(-1.,1.,13)
						nv	= np.nan_to_num(ds3['dX_dy_dpT2'].value)*pT\
							  /self.approx[specie](ds3['pT'].value)
						self.spectra[nPDF][system][sqrts][specie] = interp2d(pT, yn, nv, kind='linear')
		f.close()

	def interp(self, nPDF, system, sqrts, specie, pTs, ys):
		M = self.mass[specie]
		results = []
		for (pT, y) in zip(pTs, ys):
			mT = np.sqrt(pT**2 + M**2)
			if 2.*mT >= float(sqrts):
				results.append(0.)
				continue
			ymax = np.arccosh(float(sqrts)/2./mT)
			if np.abs(y)>=ymax:
				results.append(0.)
				continue
			yn = y/ymax	
			results.append(self.spectra[nPDF][system][sqrts][specie](pT, yn)\
			   *self.approx[specie](pT))
		return np.array(results)


