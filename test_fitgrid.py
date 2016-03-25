from starkit.gridkit.io.phoenix.base import PhoenixSpectralGridIO, ParameterSet
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest
from starkit import assemble_model
from starkit.fitkit import priors
from specutils.io import read_fits
import numpy as np
from starkit import Spectrum1D
from pylab import *

#s=read_fits.read_fits_spectrum1d('NE_1_003.fits',dispersion_unit='Angstrom')
#s=read_fits.read_fits_spectrum1d('E5_1_004.fits',dispersion_unit='Angstrom')
#s=read_fits.read_fits_spectrum1d('E5_2_001.fits',dispersion_unit='Angstrom')
s=read_fits.read_fits_spectrum1d('S0-2_080516.fits',dispersion_unit='Angstrom')
#s=read_fits.read_fits_spectrum1d('S0-2_supercombo.fits',dispersion_unit='Angstrom')

wave_range = [21000,22910]

good = np.where((s.wavelength.value > wave_range[0]) & (s.wavelength.value < wave_range[1]))[0]
s = Spectrum1D.from_array(s.wavelength[good],s.flux[good]/np.median(s.flux[good]), dispersion_unit = s.wavelength.unit)
#s = Spectrum1D.from_array(s.wavelength,s.flux/np.median(s.flux),dispersion_unit=s.wavelength.unit)
snr = 40
s.uncertainty = (np.zeros(len(s.flux.value))+1.0/snr)*s.flux.unit
plot(s.wavelength,s.flux)
g=load_grid('phoenix_20000_hk.h5')
#g=load_grid('phoenix_20000_k.h5')   # has alpha = 0
#my_model = assemble_model(g, vrad=0, vrot=0,R=5400,spectrum=s,normalize_npol=4)

# for S0-2
my_model = assemble_model(g, vrad=0, R=4000, spectrum=s,normalize_npol=1,vrot=100)

wave,flux = my_model()
clf()
plot(s.wavelength,s.flux)
#plt.plot(wave,flux)

teff_prior = priors.UniformPrior(7000,12000) # for S0-2
logg_prior = priors.UniformPrior(2.0,4.4)  # for S0-2
#mh_prior = priors.UniformPrior(-1.5,1.0)
mh_prior = priors.FixedPrior(0.0)
#alpha_prior = priors.UniformPrior(-0.2,1.2)
alpha_prior = priors.FixedPrior(0)


#vrot_prior = priors.UniformPrior(0,20)
vrot_prior = priors.UniformPrior(50,300)  # note, limb darkening is fixed, so we don't need a prior

vrad_prior = priors.UniformPrior(-800,800)
R_prior = priors.FixedPrior(4000)
#R_prior = priors.UniformPrior(4000,12000)

ll = Chi2Likelihood(s)
fit_model = my_model | ll

fitobj = MultiNest(fit_model, [teff_prior, logg_prior, mh_prior, alpha_prior,vrot_prior, vrad_prior,  R_prior])
result = fitobj.run(verbose=True,clean_up=False)

fit_wave,fit_flux = fit_model[:-1]()

plot(fit_wave,fit_flux)

result.calculate_sigmas(1)
