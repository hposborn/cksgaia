import numpy as np
# import astropy.constants as C
from astropy import constants as C
import astropy.units as u
import pandas as pd
import pylab as pl

from . import io
from . import fitting
from .config import *


def fit_cdpp(kicselect):
    """Fit polynomial to CDPP values

    Args:
        kicselect (DataFrame): DataFrame of KIC catalogue. CDPP columns should be arrays
                               with value > 0 for quarters the star was observed.

    Returns:
        DataFrame : same as input with several columns added
    """

    CDPP3s=[]
    CDPP6s=[]
    CDPP12s=[]
    for kic in kicselect.iterrows():
        CDPP3s+=[np.array(kic[1]['CDPP3'][1:-1].replace('\n','').split()).astype(float)]
        CDPP6s+=[np.array(kic[1]['CDPP6'][1:-1].replace('\n','').split()).astype(float)]
        CDPP12s+=[np.array(kic[1]['CDPP12'][1:-1].replace('\n','').split()).astype(float)]
    kicselect['CDPP3']=CDPP3s
    kicselect['CDPP6']=CDPP6s
    kicselect['CDPP12']=CDPP12s

    kicselect['m17_nquarters'] = kicselect['CDPP3'].apply(lambda x: np.clip(np.sum(x > 0), 0, 17))
    kicselect['m17_tobs'] = kicselect['m17_nquarters'].values * 90

    kicselect['m17_cdpp3'] = kicselect['CDPP3'].apply(lambda x: np.median(x[x > 0]))
    kicselect['m17_cdpp6'] = kicselect['CDPP6'].apply(lambda x: np.median(x[x > 0]))
    kicselect['m17_cdpp12'] = kicselect['CDPP12'].apply(lambda x: np.median(x[x > 0]))

    # c3 = np.ma.masked_values(np.vstack(kicselect.as_matrix(columns=['CDPP3'])[:,0]), 0.0)
    # c6 = np.ma.masked_values(np.vstack(kicselect.as_matrix(columns=['CDPP6'])[:,0]), 0.0)
    # c12 = np.ma.masked_values(np.vstack(kicselect.as_matrix(columns=['CDPP12'])[:,0]), 0.0)
    # kicselect['m17_cdpp3'] = np.median(c3, axis=1)
    # kicselect['m17_cdpp6'] = np.median(c6, axis=1)
    # kicselect['m17_cdpp12'] = np.median(c12, axis=1)

    cdpp_arr = kicselect.loc[:,['m17_cdpp3', 'm17_cdpp6', 'm17_cdpp12']].values.T
    pfit = np.polyfit(1 / np.sqrt([3., 6., 12.]), cdpp_arr, 2)

    kicselect['m17_cdpp_fit0'] = pfit[2, :]
    kicselect['m17_cdpp_fit1'] = pfit[1, :]
    kicselect['m17_cdpp_fit2'] = pfit[0, :]

    return kicselect


def detection_prob(prad, per, kicselect, nkic=None, step=False):
    if nkic is None:
        nkic = kicselect['id_kic'].count()

    # Put this planet around all other stars
    # Calculate new durations
    a = ((C.G * kicselect['m17_smass'].values * u.Msun * ((per*u.day) / (2 * np.pi)) ** 2) ** (1 / 3.))
    R = (kicselect['gaia2_srad'].values * u.Rsun)
    durations = (((per*u.day).to(u.hour).value) / np.pi) * np.arcsin((R / a).decompose().value)
    rors = ((prad * u.Rearth) / (kicselect['gaia2_srad'].values*u.Rsun)).decompose().value

    x = 1 / np.sqrt(durations)
    cdpp_durs = kicselect['m17_cdpp_fit0'].values + \
                kicselect['m17_cdpp_fit1'].values * x + \
                kicselect['m17_cdpp_fit2'].values * x ** 2

    # Calculate SNR for other stars
    other_snr = rors ** 2 * (per / kicselect['m17_tobs'].values) ** -0.5 * (1 / (cdpp_durs * 1e-6))

    # s = cksrad.fitting.logistic(other_snr, step=step)
    s = fitting.gamma_complete(other_snr, step=step)
    s[np.isnan(s)] = 0.0
    det = np.sum(s) / np.float(nkic)

    return det


def get_weights(kois, kicselect):
    """
    Add completeness columns det_prob, tr_prob, and weight into
    kois DataFrame

    Args:
        kois (DataFrame): input dataframe with candidates
        kicselect (DataFrame): DataFrame with all KIC stars in sample

    Returns:
        DataFrame: same as kois with det_prob, tr_prob, and weight columns
                   added
    """

    x = 1 / np.sqrt(kois['koi_duration'].values)
    kois['koi_cdpp_dur'] = kois['m17_cdpp_fit0'].values + \
                           kois['m17_cdpp_fit1'].values * x + \
                           kois['m17_cdpp_fit2'].values * x ** 2
    kois['koi_snr'] = kois['koi_ror'].values ** 2 * \
                      (kois['koi_period'].values / kois['m17_tobs'].values) ** -0.5 * \
                      1 / (kois['koi_cdpp_dur'].values * 1e-6)

    det_prob = []
    jobs = []
    nkic = kicselect['id_kic'].count()

    pers = kois['koi_period'].values
    prads = kois['gdir_prad'].values

    for i in range(len(pers)):
        per = pers[i]
        prad = prads[i]

        det = detection_prob(prad, per, kicselect, nkic=nkic)
        det_prob.append(det)

        # print i, per, prad, det

    det_prob = np.array(det_prob)
    # tr_prob = 0.7/kois['koi_dor'].values

    mstar_g = kois['giso_smass'].values * u.Msun
    per_s = kois['koi_period'].values * u.day

    tr_prob = 0.9 * ((kois['gdir_srad'].values * u.Rsun) / ((C.G * mstar_g * (per_s / (2 * np.pi)) ** 2) ** (1 / 3.))).decompose().value

    weights = 1 / (det_prob * tr_prob)

    kois['det_prob'] = det_prob
    kois['tr_prob'] = tr_prob
    kois['weight'] = weights

    return kois


def weight_merge(physmerge):
    kic = io.load_table('kic')
    kicselect = io.load_table('kic-filtered')
    kicselect = fit_cdpp(kicselect)

    physmerge = pd.merge(physmerge, kic, on='id_kic')

    physmerge = fit_cdpp(physmerge)
    physmerge = get_weights(physmerge, kicselect)

    return physmerge


def get_sensitivity_contour(kicselect, percentile):
    pgrid = np.logspace(np.log10(1.0), np.log10(300), 40)
    rgrid = np.logspace(np.log10(0.3), np.log10(20.0), 40)
    prob_grid = np.zeros((len(pgrid), len(rgrid)))
    sens_grid = np.zeros((len(pgrid), len(rgrid)))
    tr_grid = np.zeros((len(pgrid), len(rgrid)))

    nkic = kicselect['id_kic'].count()

    for i, p in enumerate(pgrid):
        smas = ((kicselect['m17_smass']*u.Msun * (p *u.day) ** 2) ** (1 / 3.)).to(u.AU)
        a = (C.G * (kicselect['m17_smass'] * u.Msun) * ((p * u.day) / (2 * np.pi)) ** 2) ** (1 / 3.)
        R = kicselect['gaia2_srad'] * u.Rsun
        durations = ((p * u.day) / np.pi) * np.arcsin(R / a)
        aors = (smas / (kicselect['gaia2_srad']*u.Rsun)).decompose().value

        x = 1 / np.sqrt(durations.to(u.hour))
        cdpp_dur = kicselect['m17_cdpp_fit0'] + kicselect['m17_cdpp_fit1'] * x + kicselect['m17_cdpp_fit2'] * x ** 2

        for j, r in enumerate(rgrid):
            rors = (r*u.Rearth) / (kicselect['gaia2_srad']*u.Rsun).decompose().value

            snr = ((r*u.Rearth) / (kicselect['gaia2_srad']*u.Msun)) ** 2 * (p / kicselect['m17_tobs']) ** -0.5 * (
                        1 / (cdpp_dur * 1e-6)).decompose().value

            tr = np.nanmedian((0.9 / aors))
            sens = detection_prob(r, p, kicselect, nkic=nkic, step=True)
            prob = sens * tr

            prob_grid[i, j] = prob
            sens_grid[i, j] = sens
            tr_grid[i, j] = tr


    prob_flat = prob_grid.transpose()
    sens_flat = sens_grid.transpose()
    tr_flat = tr_grid.transpose()

    colors = ['0.25', '0.25', '0.25', '0.25', '0.25', '0.25']

    CS = pl.contour(pgrid, rgrid, sens_flat, 10, levels=[0.0, percentile], colors=colors)
    v = CS.collections[1].get_paths()[0].vertices
    cx = v[:, 0]
    cy = v[:, 1]

    # level = 1.0/num_stars
    # CS = pl.contour(pgrid, rgrid, prob_flat, 10, levels=[0.0, level], colors=colors)
    # v = CS.collections[1].get_paths()[0].vertices
    # cx = v[:, 0]
    # cy = v[:, 1]

    return cx, cy
