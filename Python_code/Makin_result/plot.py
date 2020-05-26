import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('refh_results.csv')  
data = df[(df["kinematic_axis"] == "vely") &
          (df["bin_width"] == 64) &
          (df["decoder"].isin(("regression", "KF_observed", "UKF", "rEFH_dynamic")))]

plt.figure(figsize=(16,9))
sns.set(font_scale=1.4)
ax = sns.stripplot(x="decoder", y="snr", data=data, alpha=1, zorder=1)
sns.barplot(x="decoder", y="snr", data=data, alpha=0.25)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, horizontalalignment='right')
ax.set_xlabel("Decoder", fontsize=15)
ax.set_ylabel("SNR (dB)", fontsize=15)
plt.title("y-velocity", fontsize=25)
sns.despine()

plt.tight_layout()
plt.savefig('y-velocity.png')