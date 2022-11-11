import sys
import numpy as np
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ztfidr import get_sample, typing   # needs ztfquery, sncosmo
from sn_code import AnalyseSN, SNe_GP2

sns.set_style('darkgrid')


class VerticalNavigationToolbar2Tk(NavigationToolbar2Tk):
    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        b.pack(side=tk.TOP)
        return b

    def _Spacer(self):
        s = tk.Frame(self, width=26, relief=tk.RIDGE, bg='DarkGray', padx=2)
        s.pack(side=tk.TOP, pady=5)
        return s

    def set_message(self, s):
        pass


class Controller(tk.Frame):

    def __init__(self, df, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry('1200x600+10+10')
        self.master.configure(bg='#6b8ba4')
        self.master.protocol('WM_DELETE_WINDOW', self.exit_)

        self.df = df
        self.target = [None, []]
        self.px = 1 / plt.rcParams['figure.dpi']

        self.main_graph = [['dm15_g', 'abs_mag_g', 'sn_class'], [], []]
        self.param_range = {'abs_mag': [-21, -16], 'abs_mag_dr2': [-21, -16],
                            'separation': [-1, 35], 'host_mass': [7, 12], 'host_g_i': [-2, 2],
                            'Si_vel_6355': [0.8, 1.6], 'Si_vel_5972': [0.8, 1.6], 'Si_amp_6355': [0, 0.9],
                            'Si_amp_5972': [0, 0.9], 'Si_fwhm_6355': [0, 250], 'Si_fwhm_5972': [0, 250],
                            'w_6355': [0, 200], 'w_5972': [0, 60], 'W_SiII': [0, 0.6], 'R_SiII': [0, 0.8]}

        self.drop_down()
        self.get_report()
        self.update_screen()

    def exit_(self):
        self.destroy()
        sys.exit()

    def mouse_event(self, event):
        toolbar = self.main_graph[2][0]
        xx, yy, cc = self.main_graph[0]
        if not toolbar.mode.value:
            xm = float(event.xdata)
            ym = float(event.ydata)
            df_ = self.df[[xx, yy, 'ztfname']].copy().dropna()
            valx = self.param_range.get(xx, [None, None])
            valy = self.param_range.get(yy, [None, None])
            if valx[0] is None:
                maxx, minx = df_[xx].max(), df_[xx].min()
            else:
                maxx, minx = valx[1], valx[0]
            if valy[0] is None:
                maxy, miny = df_[yy].max(), df_[yy].min()
            else:
                maxy, miny = valy[1], valy[0]
            mouse = np.array([(xm - minx) / (maxx - minx), (ym - miny) / (maxy - miny)])
            coords = np.array([(df_[xx] - minx) / (maxx - minx), (df_[yy] - miny) / (maxy - miny)]).T
            calc_dist = np.vectorize(lambda i: np.linalg.norm(mouse - coords[i]))(range(len(df_)))

            self.target[0] = df_['ztfname'].iloc[calc_dist.argmin()]
            self.master.clipboard_clear()
            self.master.clipboard_append(f'{self.target[0]}')
            ind = (df_.ztfname == self.target[0])
            self.target[1] = [df_[xx][ind], df_[yy][ind]]
            self.update_screen()

    def update_main(self):
        xx, yy, cc = self.main_graph[0]
        fig, ax = self.create_fig(self.df, xx, yy, cc, figs=(600 * self.px, 500 * self.px))
        ax.set_xlabel(xx, size=15)
        ax.set_ylabel(yy, size=15)
        fig.patch.set_facecolor('#6b8ba4')
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().grid(row=3, column=0, rowspan=10, columnspan=12)
        toolbar = NavigationToolbar2Tk(canvas, self.master, pack_toolbar=False)
        toolbar.grid(row=14, column=0, rowspan=1, columnspan=12)

        if self.target[1]:
            xi, yi = self.target[1]
            ax.plot(xi, yi, color='red', marker='x', markersize=10)

        canvas.mpl_connect('button_press_event', self.mouse_event)
        self.main_graph[1] = [fig, ax]
        self.main_graph[2] = [toolbar]
        plt.tight_layout()
        plt.close(fig)

    def plot_gauss(self):
        sample = get_sample()
        if self.target[0] is not None:
            sn_test = SNe_GP2(self.target[0], sample.data, sample, (20, 30), lccf=False)
            sn_test.new_pipeline(band='g', density=100, timescale=30, noise_amp=1, recursive=True)
            sn_test.new_pipeline(band='r', density=100, timescale=30, noise_amp=1, recursive=True)
            fig, ax = sn_test.show('all', figsize=(600 * self.px, 200 * self.px))
        else:
            fig, ax = plt.subplots(figsize=(600 * self.px, 200 * self.px))
        fig.patch.set_facecolor('#6b8ba4')
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().grid(row=0, column=12, rowspan=4, columnspan=12)
        plt.close(fig)

    def plot_spec(self):
        sample = get_sample()
        fig, ax = plt.subplots(figsize=(500 * self.px, 300 * self.px))
        if self.target[0] is not None:
            spec = sample.get_target_spectra(self.target[0])
            if type(spec) == list:
                phases = [sp.get_phase() for sp in spec]
                spec_0 = spec[np.argmin(phases)]
                wl = spec_0.data['lbda']
                flux = spec_0.data['flux']
                ax.plot(wl, flux)
                ax.set_title(f'phase: {spec_0.get_phase():.2f}')
            else:
                wl = spec.data['lbda']
                flux = spec.data['flux']
                ax.plot(wl, flux)
                ax.set_title(f'phase: {spec.get_phase():.2f}')

        fig.patch.set_facecolor('#6b8ba4')
        canvas = FigureCanvasTkAgg(fig, master=self.master)
        canvas.get_tk_widget().grid(row=4, column=12, rowspan=6, columnspan=11)
        toolbar = VerticalNavigationToolbar2Tk(canvas, self.master)
        toolbar.grid(row=4, column=23, rowspan=6, columnspan=1)
        plt.close(fig)

    def create_fig(self, df, x, y, hue, figs):

        xlim = self.param_range.get(x, [-np.inf, np.inf])
        ylim = self.param_range.get(y, [-np.inf, np.inf])
        df = df[[x, y, hue]][(df[x] < xlim[1]) & (df[x] > xlim[0]) & \
                             (df[y] < ylim[1]) & (df[y] > ylim[0])]
        fig, ax = plt.subplots(figsize=figs)
        if hue == 'sn_class':
            sizes = np.ones(len(df[hue].unique())) * 70
            sizes[0] = 20
            sn_types = list(df[hue].value_counts().index)
            sn_dict = {x: i for i, x in enumerate(sn_types)}
            df = df.sort_values(by=hue, key=(lambda x: x.map(sn_dict)))
            sns.scatterplot(x=df[x], y=df[y], hue=df[hue], size=df[hue],
                            sizes=list(sizes.astype(int)), ax=ax, linewidth=0.1, edgecolor='black')
        elif hue == 'branch':
            df = df[[x, y, hue]].dropna()
            sizes = np.ones(len(df[hue].unique())) * 20
            sizes[0] = 20
            sn_types = list(df[hue].value_counts().index)
            sn_dict = {x: i for i, x in enumerate(sn_types)}
            df = df.sort_values(by=hue, key=(lambda x: x.map(sn_dict)))

            sns.scatterplot(x=df[x], y=df[y], hue=df[hue], size=df[hue],
                            sizes=list(sizes.astype(int)), ax=ax)
        else:
            clim = self.param_range.get(hue, [-np.inf, np.inf])
            df = df[[x, y, hue]][(df[hue] < clim[1]) & (df[hue] > clim[0])]
            out = ax.scatter(x=df[x], y=df[y], c=df[hue], s=10, cmap='plasma')
            ax.figure.colorbar(out)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.invert_yaxis() if y in ['abs_mag_g', 'abs_mag_g_salt', 'abs_mag_r', 'abs_mag_r_salt'] else None
        return fig, ax

    def get_report(self):
        def open_window():

            if self.target[0] is not None:
                new_root = tk.Tk()
                container = tk.Frame(new_root)
                canvas = tk.Canvas(container)
                scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)

                scrollable_frame = tk.Frame(canvas)
                scrollable_frame.bind('<Configure>',
                                      lambda event: canvas.configure(scrollregion=canvas.bbox('all')))

                canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
                canvas.configure(yscrollcommand=scrollbar.set, width=912, height=600)
                sn_typing = typing.Classifications().data
                sample = get_sample()
                stest = AnalyseSN(self.target[0], sn_typing, sample)
                fig, ax, size = stest.show(px=self.px)
                fig.patch.set_facecolor('#6b8ba4')
                display = FigureCanvasTkAgg(fig, scrollable_frame)
                display.get_tk_widget().pack()

                container.pack()
                canvas.pack(side='left', fill='both', expand=True)
                scrollbar.pack(side='right', fill='y')

                plt.close(fig)
                new_root.mainloop()

        button = tk.Button(self.master, command=open_window, text='report', bg='gray')
        button.grid(row=12, column=23, rowspan=1, columnspan=4, ipadx=16, ipady=5)

    def drop_down(self):
        options = self.df.drop('ztfname', axis=1).columns

        def drop_menu(ind):
            i_ax = tk.StringVar()
            i_ax.set(self.main_graph[0][ind])
            i_drop = tk.OptionMenu(self.master, i_ax, *options)
            i_drop.grid(row=2, column=int(1 + 3 * ind), rowspan=1, columnspan=3, ipadx=40, ipady=5)
            return i_ax

        x_ax = drop_menu(0)
        y_ax = drop_menu(1)
        c_ax = drop_menu(2)

        def update_axis():
            self.main_graph[0][0] = x_ax.get()
            self.main_graph[0][1] = y_ax.get()
            self.main_graph[0][2] = c_ax.get()
            self.target[1] = []
            self.update_main()

        button = tk.Button(self.master, command=update_axis, bg='gray')
        button.grid(row=2, column=9, rowspan=1, columnspan=4, ipadx=16, ipady=5)

    def update_screen(self):
        self.update_main()
        self.plot_gauss()
        self.plot_spec()