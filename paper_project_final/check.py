import numpy as np

fn = "out/stim/20201211/flow/crsweep-vertical-none-trial/C/flow_C_MFEFtoMLIP.npz"
Z = np.load(fn, allow_pickle=True)

t = Z["time"]
mu = Z["null_mean_AtoB"]
sd = Z["null_std_AtoB"]

print("first 30 null means:", mu[:30])
print("first 30 null stds:", sd[:30])
print("range(mu):", float(np.nanmin(mu)), float(np.nanmax(mu)))
print("range(sd):", float(np.nanmin(sd)), float(np.nanmax(sd)))
