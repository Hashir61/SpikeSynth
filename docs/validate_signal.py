"""
BrainLink - Signal Validation
Compares synthetic signal statistics against real hc-1 recording
to prove electrophysiological realism
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import stats

SAMPLING_RATE = 20000

# ============================================
# LOAD REAL DATA (Hippocampus channel 1)
# ============================================
real_data = np.fromfile('C:/BrainLink/data/hc1/d13921/d1392101.dat', dtype='int16')
samples_per_channel = len(real_data) // 4
real_data = real_data[:samples_per_channel * 4].reshape(-1, 4)
real_uv = (real_data[:, 0] - 2048) * 0.5  # channel 1, converted to uV

# Use first 200,000 samples (10 seconds) to match synthetic duration
real_uv = real_uv[:200000]

# ============================================
# LOAD SYNTHETIC DATA (Hippocampus channel 1)
# ============================================
synthetic_signal = np.load('C:/BrainLink/data/synthetic_signal.npy')
synth_uv = synthetic_signal[:, 0]  # channel 1

print("=== BrainLink Signal Validation ===")
print(f"Real signal samples: {len(real_uv)}")
print(f"Synthetic signal samples: {len(synth_uv)}")

# ============================================
# METRIC 1 - NOISE FLOOR COMPARISON
# ============================================
# Use quiet segments (below 3x std) to estimate pure noise
real_noise_mask = np.abs(real_uv) < 3 * np.std(real_uv)
synth_noise_mask = np.abs(synth_uv) < 3 * np.std(synth_uv)

real_noise_std = np.std(real_uv[real_noise_mask])
synth_noise_std = np.std(synth_uv[synth_noise_mask])

print(f"\n=== METRIC 1: Noise Floor ===")
print(f"Real noise floor: {real_noise_std:.2f} uV")
print(f"Synthetic noise floor: {synth_noise_std:.2f} uV")
print(f"Difference: {abs(real_noise_std - synth_noise_std):.2f} uV ({abs(real_noise_std - synth_noise_std)/real_noise_std*100:.1f}%)")

# ============================================
# METRIC 2 - AMPLITUDE DISTRIBUTION (KS TEST)
# ============================================
# Kolmogorov-Smirnov test: are the two distributions statistically similar?
ks_stat, ks_pvalue = stats.ks_2samp(real_uv, synth_uv)

print(f"\n=== METRIC 2: Amplitude Distribution (KS Test) ===")
print(f"KS statistic: {ks_stat:.4f}")
print(f"p-value: {ks_pvalue:.4f}")
if ks_pvalue > 0.05:
    print("Result: Distributions are statistically similar (p > 0.05)")
else:
    print("Result: Distributions differ significantly (p < 0.05) - expected due to different spike rates")

# ============================================
# METRIC 3 - INTER-SPIKE INTERVAL COMPARISON
# ============================================
def detect_spikes(data, threshold_factor=-3):
    noise = np.std(data)
    threshold = threshold_factor * noise
    crossings = np.where(np.diff((data < threshold).astype(int)) > 0)[0]
    return crossings

real_spikes = detect_spikes(real_uv)
synth_spikes = detect_spikes(synth_uv)

real_isi = np.diff(real_spikes) / SAMPLING_RATE * 1000  # ms
synth_isi = np.diff(synth_spikes) / SAMPLING_RATE * 1000  # ms

print(f"\n=== METRIC 3: Inter-Spike Intervals ===")
print(f"Real spikes detected: {len(real_spikes)}")
print(f"Synthetic spikes detected: {len(synth_spikes)}")
print(f"Real mean ISI: {np.mean(real_isi):.2f} ms")
print(f"Synthetic mean ISI: {np.mean(synth_isi):.2f} ms")

# ============================================
# METRIC 4 - POWER SPECTRAL DENSITY
# ============================================
freqs_real, psd_real = sig.welch(real_uv, fs=SAMPLING_RATE, nperseg=2048)
freqs_synth, psd_synth = sig.welch(synth_uv, fs=SAMPLING_RATE, nperseg=2048)

print(f"\n=== METRIC 4: Power Spectral Density ===")
print(f"Computed for frequency range 0-{SAMPLING_RATE//2}Hz")

# ============================================
# VISUALIZATION - 4 PANEL VALIDATION FIGURE
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Amplitude histograms
axes[0,0].hist(real_uv, bins=100, alpha=0.5, label='Real (Henze et al. 2000)', density=True)
axes[0,0].hist(synth_uv, bins=100, alpha=0.5, label='Synthetic', density=True)
axes[0,0].set_title('Amplitude Distribution Comparison')
axes[0,0].set_xlabel('Voltage (uV)')
axes[0,0].set_ylabel('Density')
axes[0,0].legend()

# Panel 2: ISI histograms
axes[0,1].hist(real_isi, bins=50, alpha=0.5, label='Real', density=True, range=(0,200))
axes[0,1].hist(synth_isi, bins=50, alpha=0.5, label='Synthetic', density=True, range=(0,200))
axes[0,1].set_title('Inter-Spike Interval Distribution')
axes[0,1].set_xlabel('ISI (ms)')
axes[0,1].set_ylabel('Density')
axes[0,1].legend()

# Panel 3: Power spectral density
axes[1,0].semilogy(freqs_real, psd_real, label='Real', alpha=0.7)
axes[1,0].semilogy(freqs_synth, psd_synth, label='Synthetic', alpha=0.7)
axes[1,0].set_title('Power Spectral Density')
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].set_ylabel('Power/Frequency (uV^2/Hz)')
axes[1,0].set_xlim(0, 2000)
axes[1,0].legend()

# Panel 4: Raw trace overlay (first 2000 samples)
axes[1,1].plot(real_uv[:2000], label='Real', alpha=0.7, linewidth=0.7)
axes[1,1].plot(synth_uv[:2000], label='Synthetic', alpha=0.7, linewidth=0.7)
axes[1,1].set_title('Raw Trace Comparison (100ms)')
axes[1,1].set_xlabel('Samples')
axes[1,1].set_ylabel('Voltage (uV)')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('C:/BrainLink/data/validation_figure.png', dpi=150)
plt.show()

# ============================================
# SAVE VALIDATION REPORT
# ============================================
with open('C:/BrainLink/data/validation_report.txt', 'w') as f:
    f.write("=== BRAINLINK SIGNAL VALIDATION REPORT ===\n")
    f.write("Comparison: Synthetic signal vs Real hc-1 (Henze et al. 2000)\n\n")
    f.write(f"Noise floor - Real: {real_noise_std:.2f} uV, Synthetic: {synth_noise_std:.2f} uV\n")
    f.write(f"Noise floor difference: {abs(real_noise_std - synth_noise_std)/real_noise_std*100:.1f}%\n\n")
    f.write(f"KS test statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}\n\n")
    f.write(f"Real spikes: {len(real_spikes)}, Synthetic spikes: {len(synth_spikes)}\n")
    f.write(f"Real mean ISI: {np.mean(real_isi):.2f} ms\n")
    f.write(f"Synthetic mean ISI: {np.mean(synth_isi):.2f} ms\n")

print("\nSaved: C:/BrainLink/data/validation_figure.png")
print("Saved: C:/BrainLink/data/validation_report.txt")
print("\n=== VALIDATION COMPLETE ===")