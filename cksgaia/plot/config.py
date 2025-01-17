# put common plotting variables and values here
import matplotlib
import pylab as pl
import os

matplotlib.rcParams['font.size'] = 12

matplotlib.rcParams['figure.figsize'] = (5, 3.5)
matplotlib.rcParams['figure.subplot.left'] = 0.15
matplotlib.rcParams['figure.subplot.bottom'] = 0.17
matplotlib.rcParams['figure.subplot.top'] = 0.97
matplotlib.rcParams['figure.subplot.right'] = 0.95
matplotlib.rcParams['grid.linewidth'] = 1.0

afs = 12  # font size for annotations

full_sample = 'cksgaia-planets'
filtered_sample = 'cksgaia-planets-weights'


modpath = '/'.join(os.path.dirname(__file__).split('/')[:-2])

print("Making plots with the {} table for the full sample and {} table for the filtered sample."
      .format(full_sample, filtered_sample))


def logscale_rad(axis='y'):
      if axis == 'x':
            pl.semilogx()
            pl.xlabel('Planet Size [Earth radii]')
            pl.xticks([.7, 1.0, 1.3, 1.8, 2.4, 3.5, 4.5, 6, 8.0])
            # pl.xticks(np.logspace(np.log10(0.7), np.log10(8), 6))
            ax = pl.gca()
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.1f'))
            pl.xlim(0.7, 6.0)
      elif axis == 'y':
            pl.semilogy()
            pl.ylabel('Planet Size [Earth radii]')
            pl.yticks([.7, 1.0, 1.3, 1.8, 2.4, 3.5, 4.5, 6, 8.0])
            # pl.xticks(np.logspace(np.log10(0.7), np.log10(8), 6))
            ax = pl.gca()
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.1f'))
            pl.ylim(0.7, 6.0)
      else:
            print("Please specify x or y axis (default: x)")

      return ax
