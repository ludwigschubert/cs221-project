#!/usr/bin/env python
"""
Show how to make date plots in matplotlib using date tick locators and
formatters.  See major_minor_demo1.py for more information on
controlling major and minor ticks

All matplotlib date plotting is done by converting date instances into
days since the 0001-01-01 UTC.  The conversion, tick locating and
formatting is done behind the scenes so this is most transparent to
you.  The dates module provides several converter functions date2num
and num2date

"""
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Bembo'

file_name = "./starcounts.txt"
r = np.loadtxt(open(file_name))

fig, ax = plt.subplots()
ax.plot(r)

ax.grid(True)
plt.xlabel('Repository Rank (inverse)')
plt.ylabel(r'Repository star count')

plot_file_name = "./starcounts.pdf"
# plt.show()
plt.savefig(plot_file_name, format='pdf')
