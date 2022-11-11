import bs4
import lxml
import math
import emcee
import lmfit
import george
import base64
import calmap
import ztfidr
import requests
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
from extinction import fm07
from datetime import datetime
from astropy import units as u
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from IPython.display import display
from scipy.optimize import minimize
from ztfidr import get_sample, typing
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
0: no warning 
1: flux_err==0 Remove unphysical errors 
2: chi2dof>3: Remove extreme outliers 
4: cloudy>1: BTS cut 
8: infobits>0: BTS cut 16: mag_lim<19.3: Cut applied in Dhawan 2021 
16: mag_lim < 19.3
32: seeing>3: Cut applied in Dhawan 2021 
64: fieldid>879: Recommended IPAC cut 
128: moonilf>0.5: Recommended IPAC cut 
256: has_baseline>1: Has a valid baseline correction 
512: airmass>2: Recommended IPAC cut 
1024: flux/flux_err>=5: Nominal detection
"""

class SNe_GP2:  # gaussian process
    def __init__(self, sn_name, sn_data_cut, sample, t_range, photo_cuts=[1, 2, 4, 8, 32, 256], lccf=False):
        
        lc_data_all = sample.get_target_lightcurve(sn_name).data
        flag_ =  np.all([(lc_data_all.flag&i_==0) for i_ in np.atleast_1d(photo_cuts)], axis=0)
        self.lc_data = lc_data_all[flag_] #  & (lc_data_all.flag != 0)
        
        mw_ebv = sn_data_cut['mwebv'][sn_name]
        mwr_v = sn_data_cut['mwr_v'][sn_name]
        init_t0 = sn_data_cut['t0'][sn_name]
        z = sn_data_cut['redshift'][sn_name]
        
        self.sn_name = sn_name
        self.A = mw_ebv * mwr_v
        self.z = z
        self.lccf_dr2 = sn_data_cut['lccoverage_flag'][sn_name] if lccf else 0
        self.init_t0 = init_t0
        self.t_range = t_range
        
        self.fl_peak = {'g': None, 'r': None}
        self.abs_mag = {'g': None, 'r': None}
        self.abs_mag_err = {'g': None, 'r': None}
        self.dm15 = {'g': None, 'r': None}
        self.dm15_err = {'g': None, 'r': None}
        self.dmm10 = {'g': None, 'r': None}
        self.dmm10_err = {'g': None, 'r': None}
        self.t0 = {'g': 0, 'r': 0}
        
        self.data = {'g': None, 'r': None} 
        self.init_phase =  {'g': None, 'r': None} 
        self.flag  = {'g': None, 'r': None} 
        self.field_id  = {'g': None, 'r': None} 
        self.flux  = {'g': None, 'r': None}
        self.flux_err  = {'g': None, 'r': None}
        
        self.gp = {'g': None, 'r': None}
        self.x_pred = {'g': None, 'r': None}
        self.lc_curve = {'g': None, 'r': None}
        self.lc_curve_err = {'g': None, 'r': None}
        self.noise_amp = {'g':1, 'r':1}
        self.current_q = {'g':[], 'r':[]}
        self.field_used = {'g':'', 'r':''}
        
        self.recursion_lim={'g': 0, 'r': 0}
        self.init_offset={'g': 0, 'r': 0}
        self.c = None
        
    def initiate_vars(self, band, recenter=False):
        if recenter:
            self.init_offset[band] = self.t0[band]
            if (band == 'g') and self.t0['g']:
                self.init_t0 += self.t0['g']
                ref_center = self.init_t0
                center = ref_center
            elif (band == 'r') and self.t0['r']:
                ref_center = self.init_t0
                center = ref_center + self.t0['r']
            else:
                center = self.init_t0
                ref_center = self.init_t0
        else:
            center = self.init_t0
            ref_center = self.init_t0
               

        rl, rr = self.t_range 
        lcdf = self.lc_data[(self.lc_data['filter'] == 'ztf'+band)]
        lcdf_cut = lcdf[(lcdf['mjd'] >= center - rl) & (lcdf['mjd'] <= center + rr)]
        self.data[band] = lcdf_cut
        self.init_phase[band] = (lcdf_cut['mjd'].values - ref_center) / (1 + self.z)
        self.flag[band] = lcdf_cut['flag'].values
        self.field_id[band] = lcdf_cut['field_id'].values
        self.flux[band] = lcdf_cut['flux'].values
        self.flux_err[band] = lcdf_cut['flux_err'].values
        
        self.gp[band] = None
        self.x_pred[band] = None
        self.lc_curve[band] = None
        self.lc_curve_err[band] = None
        self.noise_amp[band] = 1
        self.current_q[band] = []
        self.field_used[band] = ''
        
        
    
    def gaussian_process(self, timescale, band, noise_amp=1):
        x, y, yerr = self.init_phase[band], self.flux[band], self.flux_err[band]*noise_amp
        kernel = np.var(y) * george.kernels.Matern32Kernel(timescale)
        gp = george.GP(kernel, white_noise=np.log(0.1), fit_white_noise=True)
        try:
            gp.compute(x, yerr)
        except ValueError:
            return
        else:
            def likelihood(p):
                gp.set_parameter_vector(p)
                return -gp.log_likelihood(y) + gp.log_prior()

            def gradient(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y)

            try:
                result = minimize(likelihood, gp.get_parameter_vector(), jac=gradient)
            except (np.linalg.LinAlgError, ValueError) as err:
                return
            except Exception as err:
                self.gp[band] = None
                return
            else:
                gp.set_parameter_vector(result.x)
                self.gp[band] = gp
                

    def generate_curve(self, density, band, use_range=False):
        if self.gp[band] is None:
            return
        x, y = self.init_phase[band], self.flux[band]
        min_x = min(x) if not use_range else -self.t_range[0]
        max_x = max(x) if not use_range else self.t_range[1]
        x_pred = np.linspace(min_x, max_x, density)
        pred_new, pred_var_new = self.gp[band].predict(y, x_pred, return_var=True)
        xi, yi = x_pred, pred_new
        mp = yi.argmax()
        if (xi[mp] == xi[0]) or (np.std(yi) < 0.1) or (mp < 4):
            self.gp[band] = None
            return
        x_r, y_r = xi[mp-4: mp+5], yi[mp-4: mp+5]
        params = np.polyfit(x_r, y_r, 3)
        func = np.poly1d(params)
        self.t0[band] = min(func.deriv().roots, key=lambda r: abs(r - xi[mp]))
        self.fl_peak[band] = func(self.t0[band])
        
        self.x_pred[band] = x_pred
        self.lc_curve[band] = pred_new
        self.lc_curve_err[band] = pred_var_new

    @staticmethod
    def flux2mag(flux):
        with np.errstate(divide='ignore', invalid='ignore'):
            return -2.5 * np.log10(flux) + 30.0
        
    def flux2absmag(self, band, extinct=True):
        app_mag = self.flux2mag(self.lc_curve[band])
        distmod = Planck15.distmod(self.z).value
        eff_wl = {'g': 4722.74, 'r': 6339.61, 'i': 7886.13}
        A_band = fm07(np.array([eff_wl[band]]), self.A) if extinct else [0]
        return app_mag - distmod - A_band[0]
    
    @staticmethod
    def get_flags(combined_flag):
        binary_flag = np.array(list(bin(combined_flag)[2:])).astype(int)
        return 2**(len(binary_flag) - 1 - np.where(binary_flag == 1)[0])
        

    def calculate_parameters(self, band):
        if (self.gp[band] is None) or (self.field_used[band] == 'insufficient data'):
            return

        x, y, yerr = self.init_phase[band], self.flux[band], self.flux_err[band]
        x_pred, pred_new, pred_var_new = self.x_pred[band], self.lc_curve[band], self.lc_curve_err[band]
        gp = self.gp[band]
        t0 = self.t0[band]
    
        fl_peak = self.fl_peak[band]
        fl_15 = gp.predict(y, self.t0[band] + 15, return_var=True)
        fl_m10 = gp.predict(y, self.t0[band] -10, return_var=True)

        mag_peak = self.flux2mag(fl_peak)
        mag_15 = self.flux2mag(fl_15[0])
        mag_m10 = self.flux2mag(fl_m10[0])
        dm15 = mag_15 - mag_peak
        dmm10 = mag_m10 - mag_peak

        eff_wl = {'g': 4722.74, 'r': 6339.61}
        mag_extinct = fm07(np.array([eff_wl[band]]), self.A)
        dist_mod = Planck15.distmod(self.z).value
        abs_mag = mag_peak - mag_extinct[0] - dist_mod
        
        self.abs_mag[band] = abs_mag
        self.dm15[band] = dm15[0]
        self.dmm10[band] = dmm10[0]

    def show(self, sband='all', figsize=(10, 6), show_lines=True):
        fig, ax = plt.subplots(figsize=figsize)

        def band_plot(band, ax):
            x, y, yerr = self.init_phase[band], self.flux[band], self.flux_err[band]
            x_pred, pred_new, pred_var_new = self.x_pred[band], self.lc_curve[band], self.lc_curve_err[band]

            ax.fill_between(x_pred, pred_new - np.sqrt(pred_var_new), pred_new + np.sqrt(pred_var_new),
                            color=band, alpha=0.2, label=r'1 $\sigma$')
            ax.plot(x_pred, pred_new, band, lw=1.5, alpha=0.5)
            ax.errorbar(x, y, yerr, fmt=band + '.', markersize=10, label=f'noise: x{self.noise_amp[band]}')

            init_t0, t0 = self.init_t0, self.t0[band]
            if show_lines and (t0 is not None):
                ax.axvline(x=t0 - self.t0['g'], color=band, linestyle=':')
                ax.axvline(x=self.t0[band] + 15 - self.t0['g'], color=band, linestyle='--')

            ax.set_title(f"{self.sn_name}, band={sband}, g={self.field_used['g']}, r={self.field_used['r']}")
            ax.set_xlim([-self.t_range[0] - 5, self.t_range[1] + 5])
            ax.set_ylim([0, max(pred_new) * 1.2])
            ax.legend()

        if (sband == 'all') and (self.gp['g']) and (self.gp['r']):
            band_plot('g', ax)
            band_plot('r', ax)
            ax.set_ylim([0, max(max(self.lc_curve['g']) * 1.2, max(self.lc_curve['r']) * 1.2)])
        else:
            if (sband == 'all'):
                if (self.gp['g']):
                    band_plot('g', ax)
                elif (self.gp['r']):
                    band_plot('r', ax)
                else:
                    print('no plot')
            else:
                band_plot(sband, ax)
        return fig, ax
        

    def band_props(self):
        if self.abs_mag['g'] and self.abs_mag['r']:
            self.c = self.abs_mag['g'] - self.abs_mag['r']
            
    def is_64(self, band):
        x = self.init_phase[band]
        y = self.flux[band]
        y_err = self.flux_err[band]
        cond = self.field_id[band] > 881
        return [x[~cond], y[~cond], y_err[~cond], x[cond], y[cond], y_err[cond]]
    
    def lc_coverage(self, band, mjd, recenter):
        if self.lccf_dr2 == 1:
            return True
        cs = self.t0[band] if recenter and (band=='r') and self.t0[band] else 0
        mjd = mjd.copy() - cs
        left = mjd[mjd < 0]
        right = mjd[mjd > 0]
        if np.any(left) and np.any(right):
            cond1 = right.min() - left.max() <= 8
            cond2 = (len(left) >= 3) and (len(right) >= 3)
            cond3 = len(mjd[mjd > 10]) >= 1
            cond4 = len(mjd[mjd < -7.5]) >= 1
            return cond1 and cond2 and cond3 and cond4
        else:
            return False
            
    def fit_quality(self, band, diff2):
        if not self.gp[band]:
            return False
        x, y, yr = self.init_phase[band], self.flux[band], self.flux_err[band]
        ygp, yrgp = self.gp[band].predict(y, x, return_var=True)
        xx, yy, yyr = self.x_pred[band], self.lc_curve[band], self.lc_curve_err[band]
        rss = np.sum((ygp - y)**2)
        g1 = np.gradient(yy)
        g2 = np.gradient(g1)
        g1_peaks = np.where(np.diff(np.sign(g1)))[0]
        g2_peaks = np.where(np.diff(np.sign(g2)))[0]
        xp = xx[g1_peaks]
        peak_times = xp[(xp > -10) & (xp < 10)]
        cond1 = len(peak_times) == 1
        
        xp2 = xx[g2_peaks]
        peak_times2 = xp2[(xp2 > -5) & (xp2 < 5)]
        cond2 = len(peak_times2) == 0 if diff2 else True
        
        return cond1 and cond2
            
    def lc_quality(self, band, recenter=False):
        if not self.lc_coverage(band, self.init_phase[band], recenter):
            return 'insufficient data'
        
        def boost_err(errors):
            for i in range(len(errors)):
                flags = self.get_flags(self.flag[band][i])
                if np.any(np.intersect1d(flags, [128, 512])):
                    if self.flux[band][i] < 1000:
                        errors[i] *= 10
                    else:
                        errors[i] *= 1.5
            return errors
        
        x1, y1, yr1, x2, y2, yr2 = self.is_64(band)
        lccf1 = self.lc_coverage(band, x1, recenter)
        lccf2 = self.lc_coverage(band, x2, recenter)
        if lccf1 and lccf2:
            self.init_phase[band] = x1 
            self.flux[band] = y1
            self.flux_err[band] = boost_err(yr1)
            return 'IPAC'
        elif not lccf1 and not lccf2:
            return 'insufficient data'
        else:
            self.init_phase[band] = x1 if lccf1 else x2
            self.flux[band] = y1 if lccf1 else y2
            self.flux_err[band] = boost_err(yr1) if lccf1 else boost_err(yr2)
            return 'IPAC' if lccf1 else 'non-IPAC'
    
    def new_pipeline(self, band, density, timescale, noise_amp=1, 
                     recursive=False, use_range=False, recenter=False):
        self.initiate_vars(band, recenter=recenter)
        self.field_used[band] = self.lc_quality(band, recenter=True)
        if (self.field_used[band] == 'insufficient data') and recenter:
            return
        
        self.noise_amp[band] = noise_amp
        recursion_lim = 5 if recursive else noise_amp
        
        def fitting_process(band, diff2):
            while self.noise_amp[band] <= recursion_lim:
                self.gaussian_process(timescale=timescale, band=band, noise_amp=self.noise_amp[band])
                self.generate_curve(density=100, band=band, use_range=use_range)
                if self.fit_quality(band, diff2):
                    break
                else:
                    self.noise_amp[band] += 1
            else:
                if diff2:
                    self.noise_amp[band] = noise_amp
                    fitting_process(band, diff2=False)
                else:
                    self.noise_amp[band] = 1
                    self.gaussian_process(timescale=timescale, band=band, noise_amp=1)
                    self.generate_curve(density=100, band=band, use_range=use_range)
                    
        fitting_process(band, diff2=True)
        if self.t0['g'] and (self.recursion_lim[band] < 3):
            c1 = (band == 'g') and (abs(self.t0['g']) > 0.1)
            c2 = (self.field_used[band] == 'insufficient data')
            c3 = (band == 'r') and (not recenter)
            if c2 or (c1 or c3):
                self.recursion_lim[band] += 1
                self.new_pipeline(band, density, timescale, noise_amp=noise_amp, 
                                  recursive=recursive, use_range=use_range, recenter=True)
        self.generate_curve(density=density, band=band, use_range=use_range)
        self.calculate_parameters(band)


class AnalyseSN:    # web scarp supernova info
    def __init__(self, sn_name, sn_new, sample):
        sn_old_ = sample.data.loc[sn_name]
        self.z = sn_old_['redshift']
        self.lccf = sn_old_['lccoverage_flag']
        self.t0 = sn_old_['t0']
        self.sn_name = sn_name
        self.iau = sn_old_['iau_name']
        
        sn_new_ = sn_new[sn_new['target_name'] == sn_name]
        self.sn_new_class = np.array(sn_new_['value'])
        self.user_ids = np.array(sn_new_['user_id'])
        self.dates = [datetime.fromisoformat(dt) for dt in sn_new_.date_added]
        self.lc = sample.get_target_lightcurve(self.sn_name)
        self.sc = sample.get_target_spectra(self.sn_name)
        
        self.image_list=[]
         
    def get_spectra_images(self):
        login_url = 'https://typingapp.in2p3.fr/login'
        target_url = f'https://typingapp.in2p3.fr/target/{self.sn_name}'
        headers = {'User-Agent': 
                   'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)' +\
                   'Chrome/102.0.0.0 Safari/537.36'}
        payload = {'username': 'senzelr', 'password': '*********', 'submit': 'Submit'}
        with requests.Session() as session:
            res = session.get(login_url)
            get_login_html = bs4.BeautifulSoup(res._content, 'lxml')
            payload['csrf_token'] = get_login_html.find('input', id='csrf_token')['value']
            login = session.post(login_url, data=payload)
            target_html = session.get(target_url, headers=headers)
        soup = bs4.BeautifulSoup(target_html.text, "lxml")
        raw_b64 = soup.select('img')
        
        for img in range(len(raw_b64)):
            raw_data = str(raw_b64[img])[32:-3]
            image = Image.open(BytesIO(base64.b64decode(raw_data)))
            self.image_list.append(image) 
    
    def show(self, px=None):
        self.get_spectra_images()
        n_images = len(self.image_list)
        y_size = 6*(n_images+1)
        figsize = (912*px, y_size*57*px) if (px is not None) else (16, y_size)
        fig, ax = plt.subplots(nrows=n_images+1, figsize=figsize)
        
        for img in range(n_images):
            ax[img+1].imshow(self.image_list[img])
            ax[img+1].axis('off')
        
        if list(self.user_ids):
            row1 = list(self.user_ids)
            row2 = list(self.sn_new_class)
            row3 = [' '.join(dt.strftime('%c').split()[:3]) for dt in self.dates]
            ax[0].axis('off')
            table = ax[0].table(list(zip(row1, row2, row3)), 
                              colLabels=[r'$\bf{User ID}$', r'$\bf{New\:class}$', r'$\bf{Date}$'],
                              cellLoc='center', loc='center')
            table.scale(0.9, 3)
            table.set_fontsize(15)
                                                                               
        fig.suptitle(f'{self.sn_name}, {self.iau}', size=15)
        plt.tight_layout()
        return fig, ax, y_size
    
        
class AnalyseUser:  # show stats of TypingApp users
    def __init__(self, user_id, sn_new):
        df = sn_new[sn_new['user_id'] == user_id]
        self.user_id = user_id
        self.df = df
        self.sn_class = df.value
        self.sn_name = df.target_name
        self.dates =  [datetime.fromisoformat(dt) for dt in df.date_added]

    
    def show_avg_week(self, ax):
        days = [dt.strftime('%a') for dt in self.dates]
        wmap = {'Mon':0, 'Tue':1, 'Wed':2, 'Thu':3, 'Fri':4, 'Sat':5, 'Sun':6}
        days = pd.Series(days).sort_values(key=lambda x: x.map(wmap))
        self.days= days
        sns.histplot(days, ax=ax)
        ax.set_xlabel('days')
    
    def show_avg_day(self, ax):
        hours = [dt.hour for dt in self.dates]
        hours = pd.Series(hours)
        self.hours = hours
        sns.histplot(hours, ax=ax, discrete=True, color='red')
        ax.set_xticklabels([f'{int(hr)}:00' for hr in ax.get_xticks()])
        ax.set_xlabel('time')
    
    def show_git(self, fig, ax):
        dates = [dt.date() for dt in self.dates]
        df = pd.DataFrame(pd.Series(dates).value_counts())
        df.index = pd.to_datetime(df.index)
        df.columns=['vals']
        self.year=df
        cm =calmap.yearplot(df.vals, cmap='YlGn', fillcolor='lightgrey',daylabels='MTWTFSS',
                            dayticks=[0, 2, 4, 6], linewidth=2, ax=ax, monthly_border=True,
                            monthlabels=['Mar', 'Apr', 'May', 'Jun'], monthticks=[2, 3, 4, 5])
        divider = make_axes_locatable(cm)
        lcax = divider.append_axes('right', size='2%', pad=0.5)
        fig.colorbar(cm.get_children()[1], ax=ax, cax=lcax)
        ax.set_xlabel('2022')
    
    def pie(self, ax):
        labels= self.sn_class.value_counts().index
        ax.pie(self.sn_class.value_counts(), autopct='%1.1f%%', textprops={'fontsize': 15})
        ax.legend(labels=labels, loc=3)
        
    def show(self):
        fig, ax = plt.subplots(nrows=4, figsize=(15, 20))
        self.show_git(fig, ax[0])
        self.pie(ax[1])
        self.show_avg_day(ax[2])
        self.show_avg_week(ax[3])
        fig.suptitle(f'User: {self.user_id}, Total: {len(self.sn_class)}', size=15)
        plt.tight_layout()
        
        
class TypingPipeline:   # filter typing app data
    def __init__(self, typing_data, sample, cuts):
        """
        :param typing_data: unfiltered typingapp data
        :param sample: ztfidr.get_sample()
        """
        
        self.z_cuts = cuts[0]
        self.phase_cuts = cuts[1]
        self.typing_data = typing_data
        self.names = sample.data.index
        self.sn_group = self.group_type()
        self.sn_set = self.set_type()
        self.phases = self.get_phase(sample)
        
        self.targets = self.cuts(sample)
        self.sn_class = self.filter(typing_data)
        
    def group_type(self):
        master_list = []
        df = self.typing_data[['target_name', 'value']]
        for sn_name in self.names:
            sn_list = []
            dfi = df[df['target_name'] == sn_name].values.T[1]
            for i in range(len(dfi)):
                sn_list.append(dfi[i])
            for n in range(9-len(sn_list)):
                sn_list.append('0')
            master_list.append(sn_list)
            
        columns = [f'c{i}' for i in range(1, len(master_list[0])+1)]
        return pd.DataFrame(master_list, index=self.names, columns=columns)
    
    def set_type(self):
        sn_set = self.sn_group.apply(lambda x: pd.Series(sorted(list(set(x)))[::-1]), axis=1)
        sn_set = sn_set.replace(np.NaN, '0')
        sn_set.columns = [f'c{i}' for i in range(1, len(sn_set.iloc[0])+1)]
        return sn_set.drop(sn_set.columns[-1], axis=1)
    
    def get_phase(self, sample):
        warnings.filterwarnings('ignore')
        sn_phases = np.zeros(len(self.names))
        for i, sn_name in enumerate(self.names):
            spec_i = sample.get_target_spectra(sn_name)
            if type(spec_i) == list:
                sn_phases[i] = min([sc.get_phase() for sc in spec_i], key=lambda x: abs(x))
            elif type(spec_i) == ztfidr.spectroscopy.Spectrum:
                sn_phases[i] = spec_i.get_phase()
            else:
                sn_phases[i] = None
        return pd.Series(sn_phases, index=self.names)
    
    def cuts(self, sample):
        """ 
        cuts to data:
        z < 0.1
        lccf = 1
        -15 days < phase < +10 days
        """
        sn_data = sample.data
        sn_data['phase'] = self.phases
        
        sn_cut = (sn_data['redshift'] > self.z_cuts[0]) & (sn_data['redshift'] <= self.z_cuts[1])\
                  & (sn_data['phase'] >= self.phase_cuts[0]) & (sn_data['phase'] <= self.phase_cuts[1])
#                   & (sn_data['lccoverage_flag'] == 1)
        return sn_data.index[sn_cut]
    
    def filter(self, td):
        sn_class = []
        flag_snia = [8, 9, 26, 36, 29, 14]
        for sn_i in self.targets:
            c1, c2, c3, c4 = self.sn_set.c1[sn_i], self.sn_set.c2[sn_i], \
                                self.sn_set.c3[sn_i], self.sn_set.c4[sn_i]
            counts = self.sn_group.loc[sn_i].replace('0', np.nan).value_counts(dropna=True)
            users = td.user_id[(td['target_name'] == sn_i) & (td['value'] == 'sn ia')].values
            if c2 == '0':
                if c1 in ['not ia', 'ii', 'ib/c']:
                    sn_class.append('non ia')
                elif c1 == '0':
                    sn_class.append('None')
                else:
                    sn_class.append(c1)
            elif 'ia-other' in [c1, c2, c3]:
                sn_class.append('ia-other')
            elif c3 == '0':
                if (c1 == 'sn ia') & ((c2 == 'ia-norm') | (c2 == 'ia-91t') | (c2 == 'ia-91bg')):
                    if np.any(np.intersect1d(flag_snia, users)):
                        if counts['sn ia'] == 1:
                            sn_class.append(c2)
                        else:
                            if counts.index[counts.argmax()] == c2:
                                sn_class.append(c2)
                            else:
                                sn_class.append('sn ia')
                    else:
                        sn_class.append(counts.index[counts.argmax()])
                elif (c1 == 'unclear') & \
                ((c2 == 'sn ia') | (c2 == 'ia-norm') | (c2 == 'ia-91t') | (c2 == 'ia-91bg')):
                    if counts.index[counts.argmax()] == c2:
                        sn_class.append('ia-norm')
                    else:
                        sn_class.append('sn ia')
                elif len(np.intersect1d(['unclear', 'not ia', 'ii', 'ib/c'], [c1, c2])):
                    sn_class.append('non ia')
                else:
                    sn_class.append('unclear')
                    
            elif c4 == '0':
                if (c1 == 'unclear') & (c3 == 'sn ia') & (c4 == 'ia-norm'):
                    if counts.index[counts.argmax()] == 'ia-norm':
                        sn_class.append('ia-norm')
                    else:
                        sn_class.append('sn ia')
                else:
                    sn_class.append('unclear')
            else:
                sn_class.append('unclear')
            
        return pd.Series(sn_class, index=self.targets)


class FitSpectra:   # fit SiII 6355 & 5972
    def __init__(self, sn_name, sample, bounds_1, bounds_2, sigma, recursion=[False, False]):
        self.name = sn_name
        self.redshift = sample.data['redshift'][self.name]
        spec_object = sample.get_target_spectra(self.name)
        phase, data = self.get_object(spec_object)
        self.phase = phase
        self.sn_data = data
        
        self.wave = self.sn_data['lbda'].values / (1 + self.redshift)
        self.flux = self.sn_data['flux'].values
        self.flux_err = self.flux * 0.01
        
        self.c = 2.99792e5
        self.init_params = [-0.4, -1.0, 1.0, 25.0]
        
        self.c_s = {'6355': {'left': [], 'right': []},
                    '5972': {'left': [], 'right': []}}
        self.converged = {'6355': {'left': False, 'right': False},
                          '5972': {'left': False, 'right': False}}
        self.rest_wl = {'6355': [6347.103, 6371.359], '5972': [5957.56, 5978.93]}
        self.SiII_model = {'6355': self.calc_model(bounds_1, '6355', recursion=recursion[0]),
                           '5972': self.calc_model(bounds_2, '5972', recursion=recursion[1])}
        
        self.cut = self.constraints(sigma)
    
    def get_object(self, spec_object):
        if type(spec_object) != list:
            return spec_object.get_phase(), spec_object.data
        else:
            phases = np.array([spec.get_phase() for spec in spec_object])
            ind = np.abs(phases).argmin()
            phase = phases[ind]
            data = np.array(spec_object)[ind].data
            return phase, data
        
    def calc_model(self, v_bounds, wave, recursion):
        rest_Si_1, rest_Si_2 = self.rest_wl[wave]
        w1 = self.velocity2wavelength(v_bounds[0], rest_Si_1)
        w2 = self.velocity2wavelength(v_bounds[1], rest_Si_1)
        
        i_amp, i_vel, i_fwhm, anchor_width = self.init_params
        wi_c = np.where(((self.wave > w1 - anchor_width) & (self.wave < w1 + anchor_width)) | \
                        ((self.wave > w2 - anchor_width) & (self.wave < w2 + anchor_width)))
        
        wave_continuum = self.wave[wi_c]
        flux_continuum = self.flux[wi_c]
        flux_err_continuum = self.flux_err[wi_c]
        continuum_params = np.polyfit(wave_continuum, flux_continuum, deg=1, w=1/flux_err_continuum)

        wi = np.where((self.wave > w1) & (self.wave < w2))
        wl_cut = self.wave[wi]

        continuum_cut=continuum_params[1] + wl_cut * continuum_params[0]
        continuum = continuum_params[1] + self.wave * continuum_params[0]
        flux_cut = self.flux[wi] / continuum_cut - 1
        flux_err_cut = self.flux_err[wi] / continuum_cut

        model = Model(self.gaussian_at(rest_Si_1), prefix='g1_') + \
                Model(self.gaussian_at(rest_Si_2), prefix='g2_')
        params = Parameters()
        params.add('g1_amp', value=i_amp, max=0)
        params.add('g1_vel', value=i_vel, min=-4, max=0)
        params.add('g1_fwhm', value=i_fwhm, min=0)
        params.add('g2_amp', expr='g1_amp')
        params.add('g2_vel', expr='g1_vel')
        params.add('g2_fwhm', expr='g1_fwhm')
        out = model.fit(flux_cut, params, x=wl_cut, weights=1/flux_err_cut)
        
        def converge_spec(wave, model):
            dd = -self.gfit(wl_cut, wave=wave, params=model)
            norm = dd.max()*2
            c_sl= (dd[0]+dd[1])/norm 
            c_sr = (dd[-1]+dd[-2])/norm
            return [c_sl, c_sr]
        
        model_i = [out, wl_cut, flux_cut, continuum_cut, continuum, (w1, w2)]
        new_c_s = converge_spec(wave, model_i)
        self.c_s[wave]['left'].append(new_c_s[0]) if not self.converged[wave]['left'] else None 
        self.c_s[wave]['right'].append(new_c_s[1]) if not self.converged[wave]['right'] else None 
        
        def stop(side):
            lim = 15
            conv = self.c_s[wave]
            if side == 'both':
                return (len(conv['left']) > lim) and  (len(conv['right']) > lim)
            else:
                return len(conv[side]) > lim
        
        if recursion:
            params = {'6355': [0.03, 600], '5972': [0.05, 200]}
            lim, step = params[wave]
            conv = self.c_s[wave]
            if (conv['left'][-1] > lim) and (conv['right'][-1] > lim) and not stop('both'):
                model_i = self.calc_model(v_bounds + np.array([-step, step]), wave, recursion=recursion)
            elif (conv['left'][-1] > lim) and not stop('left'):
                self.converged[wave]['right'] = True
                model_i = self.calc_model(v_bounds + np.array([-step, 0]), wave, recursion=recursion)
            elif (conv['right'][-1] > lim) and not stop('right'):
                self.converged[wave]['left'] = True
                model_i = self.calc_model(v_bounds + np.array([0, step]), wave, recursion=recursion)
        
        return model_i
    
    def gfit(self, x, wave, which='both', params=[]):
        model= self.SiII_model[wave] if not params else params
        rest_Si_1, rest_Si_2 = self.rest_wl[wave]
        md = model[0].params
        amp1, vel1, fwhm1 = md['g1_amp'].value, md['g1_vel'].value, md['g1_fwhm'].value
        amp2, vel2, fwhm2 = md['g2_amp'].value, md['g2_vel'].value, md['g2_fwhm'].value
        g1 = self.gaussian_at(rest_Si_1)(x, amp1, vel1, fwhm1)
        g2 = self.gaussian_at(rest_Si_2)(x, amp2, vel2, fwhm2)
        if which == 'g1': return g1
        elif which == 'g2': return g2
        else: return g1 + g2
    
    
    def show(self, wave):
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
        model, wl_cut, flux_cut, continuum_cut, continuum, v_bounds = self.SiII_model[wave]
        g1, g2 = self.gfit(wl_cut, wave=wave, which='g1'), self.gfit(wl_cut, wave=wave, which='g2')
        
        ax1.plot(self.wave, self.flux)
        ax1.plot(wl_cut, (flux_cut + 1) * continuum_cut)
        ax1.plot(wl_cut, (g1 + 1) * continuum_cut, ':')
        ax1.plot(wl_cut, (g2 + 1) * continuum_cut, ':')
        ax1.plot(wl_cut, (g1 + g2 + 1) * continuum_cut)
        ax1.set_xlim(v_bounds[0] - 500, v_bounds[1] + 500)
        ax1.set_xlabel('rest wavelength')
        ax1.set_ylabel('flux')
    
        ax2.plot(self.wave, (self.flux / continuum) - 1, ':')
        ax2.plot(wl_cut, flux_cut)
        ax2.plot([v_bounds[0] - 100, v_bounds[1] + 100], [0, 0], '--')
        ax2.plot(wl_cut, g1, ':')
        ax2.plot(wl_cut, g2, ':')
        ax2.plot(wl_cut, g1 + g2)
        ax2.set_ylim(model.params['g1_amp']*2, 0.1)
        ax2.set_xlim(v_bounds[0] - 100, v_bounds[1] + 100)
        ax2.set_xlabel('rest wavelength')
        ax2.set_ylabel('normalized flux')
        
        plt.tight_layout()
        display(model)

    def velocity2wavelength(self, vel, rest_wl):
        z = -1 + np.sqrt((1 + vel / self.c) /( 1 - vel / self.c))
        return rest_wl * (1 + z)
    
    def gaussian_at(self, rest_wl):
        def gaussian(x, amp, vel, fwhm):
            z = (x - rest_wl) / rest_wl
            v =(((z + 1) ** 2 - 1.0 ) * self.c / (1 + (1 + z) ** 2)) / 1e4
            sigma = np.abs(fwhm) / (2 * np.sqrt(2*np.log(2)))
            return amp * np.exp(-(v - vel)**2 / (2 * sigma ** 2))
        return gaussian 
    
    def constraints(self, sigma):
        if sigma == 0:
            return True
        const = {'6355': None, '5972': None}
        for wave in const.keys():
            model = self.SiII_model[wave]
            y_ = self.gfit(model[1], wave=wave, which='both')
            res = abs(y_ - model[2])
            g_amp, fwhm = self.get_g12(wave)[slice(1, 3)]
            coverage = np.mean(model[1][1:] - model[1][:-1])
            if fwhm is not None:
                const[wave] = (abs(g_amp) > np.mean(res) + sigma*np.std(res)) & (fwhm > coverage*2)
            else:
                const[wave] = False
        return const
    
    def get_g12(self, wave):
        model, wl_cut, flux_cut, continuum_cut, continuum, v_bounds = self.SiII_model[wave]
        lims = [wl_cut.min(), wl_cut.max()]
        x_ = np.linspace(lims[0], lims[1], 1000)
        g12 = self.gfit(x_, wave, which='both')
        amp = g12.min()
        
        def FWHM(x, y):
            half_max = max(y) / 2
            loc = np.sign(half_max - y[0:-1]) - np.sign(half_max - y[1:])
            return x[np.where(loc < 0)[-1]] - x[np.where(loc > 0)[0]]
        
        wl = np.sum(self.rest_wl[wave])/2
        z = (x_[g12.argmin()] - wl) / wl
        v = (((z + 1) ** 2 - 1) * self.c / (1 + (1 + z) ** 2)) / 1e4
        fwhm = FWHM(x_, abs(g12))
        if (type(fwhm) == np.ndarray) and np.any(fwhm):
            fwhm = fwhm[0]
            std_dev = fwhm/(2*np.sqrt(2*np.log(2)))
            equiv_width = np.sqrt(2*np.pi) * abs(amp) * std_dev
        else:
            fwhm, equiv_width = None, None
        return v, amp, fwhm, equiv_width
    
    def show_gui(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.wave, self.flux, color='blue', label=f'phase: {self.phase:.2f}')
        for wave, col in [('6355', 'red'), ('5972', 'green')]:
            model, wl_cut, flux_cut, continuum_cut, continuum, v_bounds = self.SiII_model[wave]
            g1, g2 = self.gfit(wl_cut, wave=wave, which='g1'), self.gfit(wl_cut, wave=wave, which='g2')

            ax.plot(wl_cut, (flux_cut + 1) * continuum_cut, c=col)
            ax.plot(wl_cut, (g1 + g2 + 1) * continuum_cut, c='black', linewidth=2)
            ax.set_xlabel('rest wavelength')
            ax.set_ylabel('flux')
            v, amp, fwhm, equiv_width = self.get_g12(wave)
            
            wl_r = np.sum(self.rest_wl[wave])/2
            v_2_wl =  self.velocity2wavelength(v*1e4, wl_r)
            ax.axvline(v_2_wl, c=col, label=f'SiII vel: {abs(v):.2f}e4 km/s')
            ax.legend()
            ax.set_title(self.name)
            plt.tight_layout()
        return fig, ax
         

def plot_params(df, x, y, hue, size, xlim=[], ylim=[], figs=[15, 7], flipy=False):
    fig, ax = plt.subplots(figsize=figs)

    sizes = np.ones(len(df[hue].unique())) * 90
    sizes[0] = 30
    sn_types = list(df[hue].value_counts().index)
    sn_dict = {x: i for i, x in enumerate(sn_types)}
    df = df.sort_values(by=hue, key=(lambda x: x.map(sn_dict)))

    sns.scatterplot(x=df[x], y=df[y], hue=df[hue], size=df[size], sizes=list(sizes.astype(int)), ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xlim(xlim) if xlim else None
    ax.set_ylim(ylim) if ylim else None
    ax.invert_yaxis() if flipy else None
    return fig, ax


def get_gauss_vals(sne_list):
    abs_mag_g = np.zeros(len(sne_list))
    dm15_g = np.zeros(len(sne_list))
    dmm10_g = np.zeros(len(sne_list))
    abs_mag_r = np.zeros(len(sne_list))
    dm15_r = np.zeros(len(sne_list))
    dmm10_r = np.zeros(len(sne_list))
    c = np.zeros(len(sne_list))
    t0g = np.zeros(len(sne_list))
    t0r = np.zeros(len(sne_list))

    def set_lims(val, dmin, dmax):
        if val is None:
            return None
        if (val > dmin) and (val < dmax):
            return val
        else:
            return None

    for i, sn_object in enumerate(sne_list):
        sn_object.band_props()
        abs_mag_g[i] = set_lims(sn_object.abs_mag['g'], -21, -16)
        dm15_g[i] = set_lims(sn_object.dm15['g'], 0.01, 3)
        dmm10_g[i] = set_lims(sn_object.dmm10['g'], 0.01, 3)
        abs_mag_r[i] = set_lims(sn_object.abs_mag['r'], -23, -16)
        dm15_r[i] = set_lims(sn_object.dm15['r'], 0.01, 3)
        dmm10_r[i] = set_lims(sn_object.dmm10['r'], 0.01, 3)
        c[i] = set_lims(sn_object.c, -2, 2)
        t0g[i] = set_lims(sn_object.init_offset['g'], -20, 20)
        t0r[i] = set_lims(sn_object.t0['r'], -20, 20)

    s1 = pd.Series(abs_mag_g, name='abs_mag_g')
    s2 = pd.Series(dm15_g, name='dm15_g')
    s3 = pd.Series(dmm10_g, name='dmm10_g')
    s4 = pd.Series(abs_mag_r, name='abs_mag_r')
    s5 = pd.Series(dm15_r, name='dm15_r')
    s6 = pd.Series(dmm10_r, name='dmm10_r')
    s7 = pd.Series(c, name='c')
    s8 = pd.Series(t0g, name='t0g')
    s9 = pd.Series(t0r, name='t0r')
    
    return pd.concat([s1, s2, s3, s4, s5, s6, s7, s8, s9], axis=1)


def plot_hist(sn_group, sn_set):
    def get_unique_1(col):
        gu = sn_set[(sn_set.c2 == '0') & (sn_group[col] == '0') &
                    (sn_set.c1 != '0')]
        return len(gu)

    def get_unique_2(col):
        gu = sn_set[(sn_set.c2 != '0') & (sn_group[col] == '0') &
                    (sn_set.c1 != '0')]
        return len(gu)

    def plot_hist(hist, col, lab, ax):
        for i in range(len(hist)):
            hist[i].plot(kind='bar', ax=ax, width=1, color=col[i], label=lab[i])     
        ax.set_xticklabels(labels=np.arange(9), rotation=0)
        ax.legend(prop={'size': 15})
        ax.set_xlabel('# of classifications')

    def get_combos(conds):
        def get_unique_(col): return len(sn_set[conds & (sn_group[col] != '0')])
        vals = np.vectorize(get_unique_)(sn_group.columns)
        return pd.Series(np.pad((vals - np.roll(vals, -1))[:-1], (1, 1)))

    h1 = sn_group.apply(lambda x:  len(x)-x.value_counts()['0'])
    h1 = h1 - h1.shift(-1, fill_value=0)
    h1['zero'] = sum(sn_group.c1 == '0')
    h1.index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    h1 = h1.sort_index()

    vals = np.vectorize(get_unique_1)(sn_group.columns)
    h2 = pd.Series(np.pad((vals - np.roll(vals, 1))[1:-1], (1,2)))

    vals = np.vectorize(get_unique_2)(sn_group.columns)
    h3 = pd.Series(np.pad((vals - np.roll(vals, 1))[2:], (2, 1)))

    h4 = get_combos((sn_set.c1 == 'sn ia') & (sn_set.c2 == 'ia-norm')  
                     & (sn_set.c3 == '0'))
    h5 = get_combos((sn_set.c1 == 'unclear')  & (sn_set.c3 == '0')) \
                    & ((sn_set.c2 == 'ia-norm') | (sn_set.c2 == 'sn ia')) 
    h6 = get_combos((sn_set.c2 == 'ia-91t')  & (sn_set.c3 == '0')) \
                    & ((sn_set.c1 == 'ia-norm') | (sn_set.c1 == 'sn ia'))
    h7 = get_combos((sn_set.c2 == 'ia-91bg')  & (sn_set.c3 == '0')) \
                    & ((sn_set.c1 == 'ia-norm') | (sn_set.c1 == 'sn ia'))
    h8 = get_combos((sn_set.c1 == 'unclear') & (sn_set.c2 == 'sn ia') \
                    & (sn_set.c3 == 'ia-norm') & (sn_set.c4 == '0'))
    h9 = get_combos((sn_set.c2 == '0') & (sn_set.c1 == 'ia-norm'))
    h10 = get_combos((sn_set.c2 == '0') & (sn_set.c1 == 'sn ia'))
    h11 = get_combos((sn_set.c2 == '0') & (sn_set.c1 == 'unclear'))
    h12 = get_combos((sn_set.c2 == '0') & (sn_set.c1 == 'ia-91t'))
    h13 = get_combos((sn_set.c2 == '0') & (sn_set.c1 == 'ia-91bg'))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 15))
    plot_hist([h1, h2], ['red', 'green'], ['conflict', 'unique'], ax=ax1)
    plot_hist([h3, h4+h5+h6+h7+h8, h4+h5+h6+h7, h4+h5+h6, h4+h5, h4], 
              ['red', 'purple', 'magenta', 'orange', 'blue', 'green'], 
              ['other combos', 'unclear & sn ia & ia-norm', 'ia-91bg & (sn ia / ia-norm)',
               'unclear & (sn ia / ia-norm)', 'ia-91t & (sn ia / ia-norm)', 'sn ia & ia-norm'], ax=ax2)
    plot_hist([h2, h9+h10+h11+h12+h13, h9+h10+h11+h12, h9+h10+h11, h9+h10, h9], 
              ['black', 'magenta', 'blue', 'orange', 'red', 'green'], 
              ['other', 'ia-91bg', 'ia-91t', 'unclear', 'sn ia', 'ia-norm'], ax=ax3)
    
def get_abs_mag(app_mag, z, E_BV, R_v, extinct=True):
    distmod = Planck15.distmod(z).value
    Av = E_BV * R_v
    Ag = fm07(np.array([4722.74]), Av) if extinct else [0]
    return app_mag - distmod - Ag[0]

def separation_sn(g_ra, g_dec, sn_ra, sn_dec, redshift):
    d = Planck15.luminosity_distance(redshift)
    g_c = SkyCoord(ra=g_ra*u.degree, dec=g_dec*u.degree, distance=d, frame='icrs')
    sn_c = SkyCoord(ra=sn_ra*u.degree, dec=sn_dec*u.degree, distance=d, frame='icrs')
    return g_c.separation_3d(sn_c).value * 1e3

def galaxy_mass(g, i, z):
    distmod = Planck15.distmod(z).value
    Mi = i - distmod
    logM = 1.15 + 0.70 * (g - i) - 0.4 * Mi
    return logM