from starkit.gridkit.io.phoenix.base import PhoenixSpectralGridIO, ParameterSet
# need to change the following line for new version of starki
from starkit.fitkit.likelihoods import SpectralChi2Likelihood as Chi2Likelihood  
from starkit.gridkit import load_grid
from starkit.fitkit.multinest.base import MultiNest
from starkit import assemble_model
#from starkit.fitkit.samplers import priors
import pylab as plt
from specutils.io import read_fits
import numpy as np
from starkit import Spectrum1D
import sqlite3 as sqlite
from triangle import corner
import pandas as pd
import os
def mkdb():
    grid = PhoenixSpectralGridIO('sqlite:///phoenix.db3', base_dir='/Volumes/nyx_backup/data/phoenix/PHOENIX-ACES-AGSS-COND-2011/')
    
    grid.ingest()
def mkdb2016():
    grid = PhoenixSpectralGridIO('sqlite:///phoenix2016.db3', base_dir='/Volumes/nyx_backup/data/phoenix2016/PHOENIX-ACES-AGSS-COND-2011/')
    grid.ingest()
    
def mkgrid():

    grid = PhoenixSpectralGridIO('sqlite:///phoenix.db3',base_dir='/u/ghezgroup/data/phoenix_grid/PHOENIX-ACES-AGSS-COND-2011/')
    # note, need to run grid.ingest() once in order to create the db file, if it has not been done before. 
    params = (ParameterSet.mh>-2.0,)
    
    grid.to_hdf('phoenix_20000_hk.h5', params, 20000, (14700,24000), clobber=True)
    
def mkgrid2():

    grid = PhoenixSpectralGridIO('sqlite:///phoenix.db3',base_dir='/u/ghezgroup/data/phoenix_grid/PHOENIX-ACES-AGSS-COND-2011/')
    # note, need to run grid.ingest() once in order to create the db file, if it has not been done before.
    grid.ingest()
    params = (ParameterSet.mh>-2.0,ParameterSet.alpha==0)
    
    #grid.to_hdf('phoenix_20000_k.h5', params, 20000, (19650,24000), clobber=True)
    grid.to_hdf('phoenix_r20000_1.9-2.5_k.h5', params, 20000, (19000,25000), clobber=True)


def mkgrid_lowres():
    # make an R = 4000 K-band grid for testing purposes
    grid = PhoenixSpectralGridIO('sqlite:///phoenix2016.db3',base_dir='/Volumes/nyx_backup/data/phoenix2016/PHOENIX-ACES-AGSS-COND-2011/')
    # note, need to run grid.ingest() once in order to create the db file, if it has not been done before.
    # limit the range of the model so that the grid will be small
    params = (ParameterSet.mh>-1.0,ParameterSet.alpha==0,ParameterSet.teff < 6000, ParameterSet.teff > 3000,ParameterSet.logg < 4.5)
    
    grid.to_hdf('phoenix_4000_k_sample_grid.h5', params, 4000, (19000,25000), clobber=True)
    
def test_load_grid():
    #h5grid = load_grid('phoenix_20000_hk.h5')
    h5grid = load_grid('phoenix_r20000_1.9-2.5_k.h5')

    h5grid.teff = 5770
    h5grid.logg = 4.44
    h5grid.mh = 0.0
    h5grid.alpha = 0.0


    
def fitspec():

    s=read_fits.read_fits_spectrum1d('E5_1_004.fits',dispersion_unit='Angstrom')
    wave_range = [21000,24000]

    good = np.where((s.wavelength.value > wave_range[0]) & (s.wavelength.value < wave_range[1]))[0]
    s = Spectrum1D.from_array(s.wavelength[good],s.flux[good]/np.median(s.flux[good]), dispersion_unit = s.wavelength.unit)
    snr = 40
    s.uncertainty = (np.zeros(len(s.flux.value))+1.0/snr)*s.flux.unit

    #g=load_grid('phoenix_20000_k.h5')
    g = load_grid('phoenix_r20000_1.9-2.5_k.h5')
    my_model = assemble_model(g, vrad=0, vrot=0,R=5400,spectrum=s,normalize_npol=4)

    wave,flux = my_model()
    plt.clf()
    plt.plot(s.wavelength,s.flux)
    plt.plot(wave,flux)

    teff_prior = priors.UniformPrior(3000,8000)
    logg_prior = priors.UniformPrior(0.0,4.4)
    mh_prior = priors.UniformPrior(-1.5,1.0)
    alpha_prior = priors.UniformPrior(-0.2,1.2)


    vrot_prior = priors.UniformPrior(0,20)
    vrad_prior = priors.UniformPrior(-300,300)
    R_prior = priors.GaussianPrior(5400,600)

    ll = Chi2Likelihood(s)
    my_model = my_model | ll

    fitobj = MultiNest(my_model, [teff_prior, logg_prior, mh_prior, alpha_prior, vrot_prior, vrad_prior, R_prior])
    result = fitobj.run()
    return result

def plot_grid_points(db='phoenix2016.db3',saveplot=False):
    # plot the grid points
    connection = sqlite.connect(db)
    sql_query = 'SELECT teff,logg,mh,alpha FROM parameter_sets'
    tab = pd.read_sql_query(sql_query,connection)

    connection.close()

    arr = np.array([np.array(tab['teff']),np.array(tab['logg']),np.array(tab['mh']),tab['alpha']])
    corner(tab,plot_contours=False,labels=['Teff','log g', '[M/H]', '[alpha/Fe]'])
    if saveplot:
        fname = os.path.basename(db)
        fname = os.path.splitext(fname)
        plt.savefig(fname[0]+'.pdf')
