import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

names = ["indy_20160407_02",
"indy_20160411_01",
"indy_20160411_02",
"indy_20160418_01",
"indy_20160419_01",
"indy_20160420_01",
"indy_20160426_01",
"indy_20160622_01",
"indy_20160624_03",
"indy_20160627_01",
"indy_20160630_01",
"indy_20160915_01",
"indy_20160916_01",
"indy_20160921_01",
"indy_20160927_04",
"indy_20160927_06",
"indy_20160930_02",
"indy_20160930_05",
"indy_20161005_06",
"indy_20161006_02",
"indy_20161007_02",
"indy_20161011_03",
"indy_20161013_03",
"indy_20161014_04",
"indy_20161017_02",
"indy_20161024_03",
"indy_20161025_04",
"indy_20161026_03",
"indy_20161027_03",
"indy_20161206_02",
"indy_20161207_02",
"indy_20161212_02",
"indy_20161220_02",
"indy_20170123_02",
"indy_20170124_01",
"indy_20170127_03",
"indy_20170131_02",
"loco_20170210_03",
"loco_20170213_02",
"loco_20170214_02",
"loco_20170215_02",
"loco_20170216_02",
"loco_20170217_02",
"loco_20170227_04",
"loco_20170228_02",
"loco_20170301_05",
"loco_20170302_02"
]

dates = ["2016-4-7",
"2016-4-11",
"2016-4-11",
"2016-4-18",
"2016-4-19",
"2016-4-20",
"2016-4-26",
"2016-6-22",
"2016-6-24",
"2016-6-27",
"2016-6-30",
"2016-9-15",
"2016-9-16",
"2016-9-21",
"2016-9-27",
"2016-9-27",
"2016-9-30",
"2016-9-30",
"2016-10-5",
"2016-10-6",
"2016-10-7",
"2016-10-11",
"2016-10-13",
"2016-10-14",
"2016-10-17",
"2016-10-24",
"2016-10-25",
"2016-10-26",
"2016-10-27",
"2016-12-6",
"2016-12-7",
"2016-12-12",
"2016-12-20",
"2017-1-23",
"2017-1-24",
"2017-1-27",
"2017-1-31",
"2017-2-10",
"2017-2-13",
"2017-2-14",
"2017-2-15",
"2017-2-16",
"2017-2-17",
"2017-2-27",
"2017-2-28",
"2017-3-1",
"2017-3-2"]

dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

# Choose some nice levels
levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date
fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
ax.set(title="Indy and Loco Dataset")

markerline, stemline, baseline = ax.stem(dates, levels,
                                         linefmt="C3-", basefmt="k-",
                                         use_line_collection=True)

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dates)))

# annotate lines
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
for d, l, r, va in zip(dates, levels, names, vert):
    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),
                textcoords="offset points", va=va, ha="right")

# format xaxis with 4 month intervals
ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# remove y axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)

ax.margins(y=0.1)
plt.show()