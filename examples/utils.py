import numpy as np 
from os import environ, getenv
import sys

import matplotlib as mpl
#mpl.use('cairo')
import matplotlib.pylab as pl 
from matplotlib.colors import LogNorm 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("paper")

import torch
from torch import nn

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

EPS = 1e-4

def t2n(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    return t.to('cpu').detach().numpy()

def sanitize_mask(x):
    return x==x

class NH1(object):
    __slots__ = ['bins','_content','_sumw2']
    def __init__(self, bins=[0,1]):
        assert(len(bins) > 1)
        self.bins = np.array(bins )
        self._content = np.zeros(len(self.bins) - 1, dtype=np.float64)
        self._sumw2 = np.zeros(len(self.bins) - 1, dtype=np.float64)
    def iter(self):
        for x in range(self.bins.shape[0]-1):
            yield x
    def find_bin(self, x):
        for ix,edge in enumerate(self.bins):
            if x <= edge:
                return max(0, ix - 1)
        return len(self.bins) - 1
    def get_content(self, ix):
        return self._content[ix]
    def get_error(self, ix):
        return np.sqrt(self._sumw2[ix])
    def set_content(self, ix, val):
        self._content[ix] = val
    def set_error(self, ix, val):
        self._sumw2[ix] = val * val;
    def clear(self):
        self._content *= 0
        self._sumw2 *= 0
    def fill(self, x, y=1):
        ix = self.find_bin(x)
        self._content[ix] += y
        self._sumw2[ix] = pow(y, 2)
    def fill_array(self, x, weights=None):
        mask = sanitize_mask(x)
        mask &= sanitize_mask(weights)
        x_masked = x[mask]
        weights_masked = None if (weights is None) else weights[mask]
        w2 = None if (weights_masked is None) else np.square(weights_masked)
        hist = np.histogram(x_masked, bins=self.bins, weights=weights_masked, density=False)[0]
        herr = np.histogram(x_masked, bins=self.bins, weights=w2, density=False)[0]
        self._content += hist
        self._sumw2 += herr
    def add_array(self, arr):
        self._content += arr.astype(np.float64)
    def save(self, fpath):
        save_arr = np.array([
                self.bins, 
                np.concatenate([self._content, [0]])
            ])
        np.save(fpath, save_arr)
    def _load(self, fpath):
        load_arr = np.load(fpath)
        self.bins = load_arr[0]
        self._content = load_arr[1][:-1]
    @classmethod
    def load(x, fpath):
        if isinstance(x, NH1):
            x._load(fpath)
        else:
            h = NH1()
            h._load(fpath)
            return h
    def add_from_file(self, fpath):
        load_arr = np.load(fpath)
        try:
            assert(np.array_equal(load_arr[0], self.bins))
        except AssertionError as e:
            print(fpath)
            print(load_arr[0])
            print(self.bins)
            raise e
        add_content = load_arr[1][:-1].astype(np.float64)
        self._content += add_content
    def clone(self):
        new = NH1(self.bins)
        new._content = np.array(self._content, copy=True)
        new._sumw2 = np.array(self._sumw2, copy=True)
        return new
    def add(self, rhs, scale=1):
        assert(self._content.shape == rhs._content.shape)
        self._content += scale * rhs._content
        self._sumw2 += scale * rhs._sumw2
    def multiply(self, rhs):
        assert(self._content.shape == rhs._content.shape)
        self_rel = self._sumw2 / _clip(self._content)
        rhs_rel = rhs._sumw2 / _clip(rhs._content)
        self._content *= rhs._content 
        self._sumw2 = (np.power(self_rel, 2) + np.power(rhs_rel, 2)) * self._content
    def divide(self, den, clip=False):
        inv = den.clone()
        inv.invert()
        self.multiply(inv)
        if clip:
            self._content[den._content <= _epsilon] = 1
    def integral(self, lo=None, hi=None):
        if lo is None:
            lo = 0
        if hi is None:
            hi = self._content.shape[0]
        return np.sum(self._content[lo:hi])
    def scale(self, scale=None):
        norm = float(scale if (scale is not None) else 1./self.integral())
        self._content *= norm 
        self._sumw2 *= (norm ** 2)
    def invert(self):
        for ix in range(self._content.shape[0]):
            val = self._content[ix]
            if val != 0:
                relerr = np.sqrt(self._sumw2[ix])/val 
                self._content[ix] = 1./val
                self._sumw2[ix] = relerr * self._content[ix]
            else:
                self._content[ix] = _epsilon
                self._sumw2[ix] = 0
    def quantile(self, eff, interp=False):
        den = 1. / self.integral()
        threshold = eff * self.integral()
        for ib,b1 in enumerate(self.bins):
            frac1 = self.integral(hi=ib) 
            if frac1 >= threshold:
                if not interp or ib == 0:
                    return b1

                frac2 = self.integral(hi=(ib-1)) 
                b2 = self.bins[ib-1]
                b0 = (b1 + 
                      ((threshold - frac1) * 
                       (b2 - b1) / (frac2 - frac1)))
                return b0

    def eval_array(self, arr):
        def f(x):
            return self.get_content(self.find_bin(x))
        f = np.vectorize(f)
        return f(arr)
    def plot(self, color, label, errors=False):
        bin_centers = 0.5*(self.bins[1:] + self.bins[:-1])
        if errors and np.max(np.abs(self._sumw2)) > 0:
            errs = np.sqrt(self._sumw2)
        else:
            errs = None
        plt.errorbar(bin_centers, 
                     self._content,
                     yerr = errs,
                     drawstyle = 'steps-mid',
                     color=color,
                     label=label,
                     linewidth=2)
    def mean(self):
        sumw = 0 
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        for ix in range(bin_centers.shape[0]):
            sumw += bin_centers[ix] * self._content[ix+1]
        return sumw / self.integral()
    def median(self):
        return self.quantile(eff = 0.5)
    def stdev(self, sheppard = False):
        # sheppard = True applies Sheppard's correction, assuming constant bin-width
        mean = self.mean()
        bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        integral = self.integral()
        variance = np.sum(bin_centers * bin_centers * self._content)
        variance -= integral * mean * mean
        variance /= (integral - 1)
        if sheppard:
            variance -= pow(self.bins[1] - self.bins[0], 2) / 12 
        return np.sqrt(max(0, variance))


class Plotter(object):
    def __init__(self):
        self.hists = []
        self.ymin = None
        self.ymax = None
        self.auto_yrange = False
    def add_hist(self, hist, label='', plotstyle='b'):
        if type(plotstyle) == int:
            plotstyle = default_colors[plotstyle]
        self.hists.append((hist, label, plotstyle))
    def clear(self):
        plt.clf()
        self.hists = [] 
        self.ymin = None
        self.ymax = None
    def plot(self, xlabel=None, ylabel=None, output=None, errors=True, logy=False):
        plt.clf()
        ax = plt.gca()
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        for hist, label, plotstyle in self.hists:
            hist.plot(color=plotstyle, label=label, errors=errors)
        if xlabel:
            plt.xlabel(xlabel, fontsize=24)
        if ylabel:
            plt.ylabel(ylabel, fontsize=24)
        if logy:
            plt.yscale('log', nonposy='clip')
        plt.legend(loc=0, fontsize=20, frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if not self.auto_yrange:
            if self.ymax is not None:
                ax.set_ylim(top=self.ymax)
            if self.ymin is not None:
                ax.set_ylim(bottom=self.ymin)
            elif not logy:
                ax.set_ylim(bottom=0)
        plt.draw()
        if 'output':
            print('Creating',output)
            plt.savefig(output+'.png',bbox_inches='tight',dpi=100)
            plt.savefig(output+'.pdf',bbox_inches='tight')
        else:
            plt.show()


class Roccer(object):
    def __init__(self, y_range=range(-5,1), axis=[0.2,1,0.0005,1]):
        self.cfgs = []
        self.axis = axis
        self.yticks = [10**x for x in y_range]
        self.yticklabels = [('1' if x==0 else r'$10^{%i}$'%x) for x in y_range]
        self.xticks = [0.2, 0.4, 0.6, 0.8, 1]
        self.xticklabels = map(str, self.xticks)
    def add_vars(self, sig_hists, bkg_hists, labels, order=None):
        if order is None:
            order = sorted(sig_hists)
        try:
            for h in order: 
                try:
                    label = labels[h]
                    if type(label) == str:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label, None, '-'))
                    elif len(label) == 1:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], None, '-'))
                    elif len(label) == 2:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], label[1], '-'))
                    else:
                        self.cfgs.append((sig_hists[h], bkg_hists[h], label[0], label[1], label[2]))
                except KeyError:
                    pass # something wasn't provided - skip!
        except TypeError as e :#only one sig_hist was handed over - not iterable
            if type(labels) == str:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels, None, '-'))
            elif len(labels) == 1:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], None, '-'))
            elif len(labels) == 2:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], labels[1], '-'))
            else:
                self.cfgs.append((sig_hists[h], bkg_hists[h], labels[0], labels[1], labels[2]))
    def clear(self):
        self.cfgs = []
    def plot(self, output):
        fig, ax = plt.subplots(1)
        ax.get_xaxis().set_tick_params(which='both',direction='in')
        ax.get_yaxis().set_tick_params(which='both',direction='in')
        ax.grid(True,ls='-.',lw=0.4,zorder=-99,color='gray',alpha=0.7,which='major')

        min_value = 1

        colors = pl.cm.tab10(np.linspace(0,1,len(self.cfgs)))

        for i, (sig_hist, bkg_hist, label, customcolor, linestyle) in enumerate(self.cfgs):
            h_sig = sig_hist
            h_bkg = bkg_hist
            rmin = h_sig.bins[0]
            rmax = h_sig.bins[len(h_sig.bins)-1]

            epsilons_sig = []
            epsilons_bkg = []

            inverted = h_sig.median() < h_bkg.median()

            total_sig = h_sig.integral()
            total_bkg = h_bkg.integral()

            nbins = h_sig.bins.shape[0]
            for ib in range(nbins+1):
                if inverted:
                    esig = h_sig.integral(hi=ib) / total_sig
                    ebkg = h_bkg.integral(hi=ib) / total_bkg
                else:
                    esig = h_sig.integral(lo=ib) / total_sig
                    ebkg = h_bkg.integral(lo=ib) / total_bkg
                epsilons_sig.append(esig)
                epsilons_bkg.append(ebkg)
                if ebkg < min_value and ebkg > 0:
                    min_value = ebkg
            if customcolor is None:
                color = colors[i]
            elif type(customcolor) == int:
                color = default_colors[customcolor]
            else:
                color = customcolor
            print(customcolor)
            plt.plot(epsilons_sig, epsilons_bkg, color=color, label=label, linewidth=2, ls=linestyle)

        plt.axis(self.axis)
        ax = plt.gca()
        #plt.set_xlim(self.axis[:2])
        #plt.set_ylim(self.axis[-2:])
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        plt.yscale('log', nonposy='clip')
        plt.xscale('log', nonposx='clip')
        plt.legend(loc=4, fontsize=20, frameon=False)
        plt.ylabel('Background fake rate', fontsize=22)
        plt.xlabel('Signal efficiency', fontsize=22)
        plt.text(0.06,0.91,r'$\mathrm{H}\to\mathrm{b}\overline{\mathrm{b}}$ vs. QCD',transform=ax.transAxes,fontsize=20)
        plt.text(0.06,0.85,r'AK8, $p_\mathrm{T}>100\,\mathrm{GeV}$',transform=ax.transAxes,fontsize=20)
        ax.set_yticks(self.yticks)
        ax.set_yticklabels(self.yticklabels)
        ax.set_xticks(self.xticks)
        ax.set_xticklabels(self.xticklabels)

        print('Creating',output)
        plt.savefig(output+'.png',bbox_inches='tight',dpi=300)
        plt.savefig(output+'.pdf',bbox_inches='tight')


class METResolution(object):
    def __init__(self, bins=np.linspace(-100, 100, 40)):
        self.bins = bins
        self.bins_2 = (0, 300)
        self.bins_met1 = (0, 300)
        self.df = None
        self.df_p = None
        self.df_model = None
        self.df_truth = None
        self.df_puppi = None
        self.reset()

    def reset(self):
        self.dist = None
        self.dist_p = None
        self.dist_pup = None
        self.dist_2 = None
        self.dist_met = None
        self.dist_pred = None
        self.dist_2_p = None
        self.dist_2_pup = None

    @staticmethod
    def _compute_res(pt, phi, w, gm):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = np.sum(px, axis=-1)
        mety = np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        res = (met / gm) - 1
        return res

    def compute(self, pf, pup, gm, pred, weight=None):
        res = (pred - gm)
        res_p = (pf - gm)
        res_pup = (pup - gm)

        hist, _ = np.histogram(res, bins=self.bins)
        hist_met, self.bins_met = np.histogram(gm, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_pred, self.bins_pred = np.histogram(pred, bins=np.linspace(*(self.bins_met1) + (100,)))
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        hist_pup, _ = np.histogram(res_pup, bins=self.bins)
        hist_2, _, _ = np.histogram2d(gm, res, bins=100, range=(self.bins_2, self.bins_2))
        hist_2_p, _, _ = np.histogram2d(gm, p, bins=100, range=(self.bins_2, self.bins_2))
        hist_2_pup, _, _ = np.histogram2d(gm, pup, bins=100, range=(self.bins_2, self.bins_2))
        if self.dist is None:
            self.dist = hist + EPS
            self.dist_met = hist_met + EPS
            self.dist_pred = hist_pred + EPS
            self.dist_p = hist_p + EPS
            self.dist_pup = hist_pup + EPS
            self.dist_2 = hist_2 + EPS
            self.dist_2_p = hist_2_p + EPS
            self.dist_2_pup = hist_2_pup + EPS
        else:
            self.dist += hist
            self.dist_met += hist_met
            self.dist_pred += hist_pred
            self.dist_p += hist_p
            self.dist_pup += hist_pup
            self.dist_2 += hist_2
            self.dist_2_p += hist_2_p
            self.dist_2_pup += hist_2_pup

    @staticmethod
    def _compute_moments(x, dist):
        dist = dist / np.sum(dist)
        mean = np.sum(x * dist)
        var = np.sum(np.power(x - mean, 2) * dist)
        return mean, var

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        mean_pup, var_pup = self._compute_moments(x, self.dist_pup)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_pup:.1f}' + r'\pm' + f'{np.sqrt(var_pup):.1f})$'
        plt.hist(x=x, weights=self.dist_pup, label=label, histtype='step', bins=self.bins)

        label = r'Ground Truth ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins, linestyle='--')

        plt.xlabel('Predicted-True MET [GeV]')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        x = (self.bins_met[:-1] + self.bins_met[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_met, bins=self.bins_met)
        plt.xlabel('True MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_true.' + ext)

        plt.clf()
        x = (self.bins_pred[:-1] + self.bins_pred[1:]) * 0.5
        plt.hist(x=x, weights=self.dist_pred, bins=self.bins_pred)
        plt.xlabel('Predicted MET')
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_pred.' + ext)

        plt.clf()
        self.dist_2 = np.ma.masked_where(self.dist_2 < 0.5, self.dist_2)
        plt.imshow(self.dist_2.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True MET [GeV]')
        plt.ylabel('Predicted MET [GeV]')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr.' + ext)
        plt.clf()
        self.dist_2_p = np.ma.masked_where(self.dist_2_p < 0.5, self.dist_2_p)
        plt.imshow(self.dist_2_p.T, vmin=0.5, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.colorbar()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_corr_pf.' + ext)
        self.reset()

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}



class ParticleMETResolution(METResolution):
    @staticmethod
    def _compute_res(pt, phi, w, gm, gmphi):
        pt = pt * w
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        metx = (-1)*np.sum(px, axis=-1)
        mety = (-1)*np.sum(py, axis=-1)
        met = np.sqrt(np.power(metx, 2) + np.power(mety, 2))
        gmx = gm * np.cos(gmphi)
        gmy = gm * np.sin(gmphi)
        res =  met - gm # (met / gm) - 1                                                                                                          
        resx = metx - gmx
        resy = mety - gmy
        return res,resx,resy

    def compute(self, pt, phi, w, y, baseline, gm, gmphi):
        res,resx,resy = self._compute_res(pt, phi, w, gm, gmphi)
        res_t,resx_t,resy_t = self._compute_res(pt, phi, y, gm, gmphi)
        res_p,resx_p,resy_p = self._compute_res(pt, phi, baseline, gm, gmphi)

        #data = {'x': res, 'col_2': ['a', 'b', 'c', 'd']}                                                                                         
        df = pd.DataFrame()
        df_p = pd.DataFrame()
        df['y'] = res
        df_p['y'] = res_p
        df['x'] = gm
        df_p['x'] = gm

        bins = np.linspace(0., 300., num=25)
        df['bin'] = np.searchsorted(bins, df['x'].values)
        df_p['bin'] = np.searchsorted(bins, df_p['x'].values)

        hist, _ = np.histogram(res, bins=self.bins)
        histx, _ = np.histogram(resx, bins=self.bins)
        histy, _ = np.histogram(resy, bins=self.bins)
        hist_p, _ = np.histogram(res_p, bins=self.bins)
        histx_p, _ = np.histogram(resx_p, bins=self.bins)
        histy_p, _ = np.histogram(resy_p, bins=self.bins)
        hist_met, _ = np.histogram(res_t, bins=self.bins)
        histx_met, _ = np.histogram(resx_t, bins=self.bins)
        histy_met, _ = np.histogram(resy_t, bins=self.bins)
        hist_2, _, _ = np.histogram2d(gm, res+gm, bins=25, range=(self.bins_2, self.bins_2))
        hist_2_p, _, _ = np.histogram2d(gm, res_p+gm, bins=25, range=(self.bins_2, self.bins_2))

        self.xedges = bins

        if self.df is None:
            self.df = df
            self.df_p = df_p
        else:
            tmp = pd.concat([self.df,df],ignore_index=True,sort=False)
            tmp_p = pd.concat([self.df_p,df_p],ignore_index=True,sort=False)
 
            self.df = tmp
            self.df_p = tmp_p

        if self.dist is None:
            self.dist = hist + EPS
            self.distx = histx + EPS
            self.disty = histy + EPS
            self.dist_p = hist_p + EPS
            self.distx_p = histx_p + EPS
            self.disty_p = histy_p + EPS
            self.dist_met = hist_met + EPS
            self.distx_met = histx_met + EPS
            self.disty_met = histy_met + EPS
            self.dist_2 = hist_2 + EPS
            self.dist_2_p = hist_2_p + EPS
        else:
            self.dist += hist
            self.distx += histx
            self.disty += histy
            self.dist_p += hist_p
            self.distx_p += histx_p
            self.disty_p += histy_p
            self.dist_met += hist_met
            self.distx_met += histx_met
            self.disty_met += histy_met
            self.dist_2 += hist_2
            self.dist_2_p += hist_2_p

    def plot(self, path):
        plt.clf()
        x = (self.bins[:-1] + self.bins[1:]) * 0.5

        mean, var = self._compute_moments(x, self.dist)
        meanx, varx = self._compute_moments(x, self.distx)
        meany, vary = self._compute_moments(x, self.disty)
        mean_p, var_p = self._compute_moments(x, self.dist_p)
        meanx_p, varx_p = self._compute_moments(x, self.distx_p)
        meany_p, vary_p = self._compute_moments(x, self.disty_p)
        mean_met, var_met = self._compute_moments(x, self.dist_met)
        meanx_met, varx_met = self._compute_moments(x, self.distx_met)
        meany_met, vary_met = self._compute_moments(x, self.disty_met)

        label = r'Model ($\delta=' + f'{mean:.1f}' + r'\pm' + f'{np.sqrt(var):.1f})$'
        plt.hist(x=x, weights=self.dist, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{mean_p:.1f}' + r'\pm' + f'{np.sqrt(var_p):.1f})$'
        plt.hist(x=x, weights=self.dist_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{mean_met:.1f}' + r'\pm' + f'{np.sqrt(var_met):.1f})$'
        plt.hist(x=x, weights=self.dist_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(Predicted - True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '.' + ext)

        plt.clf()
        fig, ax = plt.subplots()

        label = r'Model ($\delta=' + f'{meanx:.1f}' + r'\pm' + f'{np.sqrt(varx):.1f})$'
        plt.hist(x=x, weights=self.distx, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{meanx_p:.1f}' + r'\pm' + f'{np.sqrt(varx_p):.1f})$'
        plt.hist(x=x, weights=self.distx_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{meanx_met:.1f}' + r'\pm' + f'{np.sqrt(varx_met):.1f})$'
        plt.hist(x=x, weights=self.distx_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(X Predicted - X True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_x.' + ext)

        plt.clf()
        fig, ax = plt.subplots()

        label = r'Model ($\delta=' + f'{meany:.1f}' + r'\pm' + f'{np.sqrt(vary):.1f})$'
        plt.hist(x=x, weights=self.distx, label=label, histtype='step', bins=self.bins)

        label = r'Puppi ($\delta=' + f'{meany_p:.1f}' + r'\pm' + f'{np.sqrt(vary_p):.1f})$'
        plt.hist(x=x, weights=self.distx_p, label=label, histtype='step', bins=self.bins)

        label = r'Truth+PF ($\delta=' + f'{meany_met:.1f}' + r'\pm' + f'{np.sqrt(vary_met):.1f})$'
        plt.hist(x=x, weights=self.distx_met, label=label, histtype='step', bins=self.bins)

        plt.xlabel('(Y Predicted - Y True)')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_y.' + ext)

        plt.clf()
        fig, ax = plt.subplots()
        plt.imshow(self.dist_2.T, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True $p_T^{miss}$ (GeV)')
        plt.ylabel('PUMA $p_T^{miss}$ (GeV)')
        plt.colorbar()
        fig.tight_layout()

        for ext in ('pdf', 'png'):
            plt.savefig(path + '_2D_puma.' + ext)

        plt.clf()
        fig, ax = plt.subplots()
        plt.imshow(self.dist_2_p.T, extent=(self.bins_2 + self.bins_2),
                   origin='lower')
        plt.xlabel('True $p_T^{miss}$ (GeV)')
        plt.ylabel('PUPPI $p_T^{miss}$ (GeV)')
        plt.colorbar()
        fig.tight_layout()

        for ext in ('pdf', 'png'):
            plt.savefig(path + '_2D_puppi.' + ext)

        return {'model': (mean, np.sqrt(var)), 'puppi': (mean_p, np.sqrt(var_p))}


class PapuMetrics(object):
    def __init__(self):
        self.loss_calc = nn.MSELoss(
            reduction='none'
        ) 
        self.reset()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.pos_acc = 0
        self.neg_acc = 0
        self.n_pos = 0
        self.n_particles = 0
        self.n_steps = 0
        self.hists = {}
        self.bins = {}

    @staticmethod
    def make_roc(pos_hist, neg_hist):
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        tp, fp = [], []
        for i in np.arange(pos_hist.shape[0], -1, -1):
            tp.append(pos_hist[i:].sum())
            fp.append(neg_hist[i:].sum())
        auc = np.trapz(tp, x=fp)
        plt.plot(fp, tp, label=f'AUC={auc:.3f}')
        return fp, tp

    def add_values(self, val, key, w=None, lo=0, hi=1):
        hist, bins = np.histogram(
                val, bins=np.linspace(lo, hi, 100), weights=w)
        if key not in self.hists:
            self.hists[key] = hist + EPS
            self.bins[key] = bins
        else:
            self.hists[key] += hist

    def compute(self, yhat, y, w=None, m=None, plot_m=None):
        #y = y.view(-1)
        #yhat = yhat.view(-1)

        #print(yhat)
        #print(yhat.shape)
        #print(y)
        #print(y.shape)
        yhat = torch.tensor(yhat)
        y = torch.tensor(y)
        loss = self.loss_calc(torch.tensor(yhat), torch.tensor(y))
        #print(loss)
        yhat = torch.clamp(yhat, 0 , 1)
        if w is not None:
            wv = w.view(-1)
            loss *= wv
        if m is None:
            m = torch.ones_like(y, dtype=bool)
        if plot_m is None:
            plot_m = m
        m = m.view(-1).float()
        plot_m = m.view(-1)
        loss *= m

        nan_mask = t2n(torch.isnan(loss)).astype(bool)

        loss = torch.mean(loss)
        self.loss += t2n(loss).mean()

        if nan_mask.sum() > 0:
            yhat = t2n(yhat)
            print(nan_mask)
            print(yhat[nan_mask])
            print()

        plot_m = t2n(plot_m).astype(bool)
        y = t2n(y)[plot_m]
        if w is not None:
            w = t2n(w).reshape(-1)[plot_m]
        yhat = t2n(yhat)[plot_m]
        n_particles = plot_m.sum()

        # let's define positive/negative by >/< 0.5                                                                                               
        y_bin = y > 0.5
        yhat_bin = yhat > 0.5

        acc = (y_bin == yhat_bin).sum() / n_particles
        self.acc += acc

        n_pos = y_bin.sum()
        pos_acc = (y_bin == yhat_bin)[y_bin].sum() / n_pos
        self.pos_acc += pos_acc
        n_neg = (~y_bin).sum()
        neg_acc = (y_bin == yhat_bin)[~y_bin].sum() / n_neg
        self.neg_acc += neg_acc

        self.n_pos += n_pos
        self.n_particles += n_particles

        self.n_steps += 1

        self.add_values(
            y, 'truth', w, -0.2, 1.2)
        self.add_values(
            yhat, 'pred', w, -0.2, 1.2)
        self.add_values(
            yhat-y, 'err', w, -2, 2)
        return loss, acc

    def mean(self):
        return ([x / self.n_steps
                 for x in [self.loss, self.acc, self.pos_acc, self.neg_acc]]
                + [self.n_pos / self.n_particles])

    def plot(self, path):
        plt.clf()
        bins = self.bins['truth']
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,                                                                                                                   
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['truth'], label='Truth', **hist_args)
        plt.hist(weights=self.hists['pred'], label='Pred', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'$E_{\mathrm{hard}}/E_{\mathrm{tot.}}$')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_e.' + ext)

        plt.clf()
        bins = self.bins['err']
        x = (bins[:-1] + bins[1:]) * 0.5
        hist_args = {
                'histtype': 'step',
                #'alpha': 0.25,                                                                                                                   
                'bins': bins,
                'log': True,
                'x': x,
                'density': True
            }
        plt.hist(weights=self.hists['err'], label='Error', **hist_args)
        plt.ylim(bottom=0.001, top=5e3)
        plt.xlabel(r'Prediction - Truth')
        plt.legend()
        for ext in ('pdf', 'png'):
            plt.savefig(path + '_err.' + ext)


