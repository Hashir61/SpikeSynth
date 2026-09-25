"""
BrainLink - Node B Signal Validation
Compares synthetic motor cortex signal statistics against 
real ALM-1 recording (Li et al. 2015)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import stats
import h5py

SAMPLING_RATE = 20000

# ============================================
# LOAD REAL DATA - extract raw waveform samples from unit_01
# ============================================
print("=== Loading real ALM-1 waveform data ===")
nwb_path = 'C:/BrainLink/data/data_structure_ANM218453_20131015.nwb'

with h5py.File(nwb_path, 'r') as f:
    # Use unit_01 waveforms as representative real spike shapes
    real_waveforms = f['processing/extracellular_units/EventWaveform/unit_06/data'][()]
    real_times = f['processing/extracellular_units/UnitTimes/unit_06/times'][()]

print(f"Real waveforms loaded: {real_waveforms.shape}")
print(f"Real spike times loaded: {len(real_times)}")

# ============================================
# LOAD SYNTHETIC DATA
# ============================================
synthetic_signal = np.load('C:/BrainLink/data/nodeB_signal.npy')
synth_uv = synthetic_signal[:, 0]  # channel 1

print(f"Synthetic signal loaded: {synth_uv.shape}")

# ============================================
# METRIC 1 - WAVEFORM SHAPE COMPARISON
# ============================================
real_mean_waveform = real_waveforms.mean(axis=0)

# Get synthetic waveforms by detecting spikes
# Use GROUND TRUTH spike times for the pyramidal neuron on channel 0
# instead of re-detecting via threshold, since detection on a mixed
# channel (pyramidal + FS) contaminates the comparison with the wrong neuron.
gt_data = np.load('C:/BrainLink/data/nodeB_ground_truth_spike_times.npz')
crossings = gt_data['neuron_0']  # neuron 0 = channel 0, pyramidal (first neuron assigned per channel)

print(f"Using ground truth spike times for neuron 0 (pyramidal, ch1): {len(crossings)} spikes")

pre, post = 14, 15
synth_waveforms = []
for idx in crossings:
    if idx > pre and idx + post < len(synth_uv):
        wf = synth_uv[idx-pre:idx+post]
        peak_offset = np.argmin(wf)
        center = idx - pre + peak_offset
        aligned_start = center - 14
        aligned_end = aligned_start + 29
        if aligned_start >= 0 and aligned_end < len(synth_uv):
            synth_waveforms.append(synth_uv[aligned_start:aligned_end])
synth_waveforms = np.array(synth_waveforms)

print(f"\n=== METRIC 1: Waveform Shape ===")
print(f"Real waveforms: {len(real_waveforms)}")
print(f"Synthetic waveforms detected: {len(synth_waveforms)}")

if len(synth_waveforms) > 0:
    synth_mean_waveform = synth_waveforms.mean(axis=0)
    # Normalize both to compare shape (not absolute scale, since units differ)
    real_norm = real_mean_waveform / np.max(np.abs(real_mean_waveform))
    if len(synth_mean_waveform) == len(real_norm):
        synth_norm = synth_mean_waveform / np.max(np.abs(synth_mean_waveform))
        correlation = np.corrcoef(real_norm, synth_norm)[0, 1]
        print(f"Waveform shape correlation: {correlation:.4f}")
        print(f"(1.0 = perfect match, closer to 1 = more realistic)")

# ============================================
# METRIC 2 - FIRING RATE COMPARISON
# ============================================
real_duration = real_times[-1] - real_times[0]
real_rate = len(real_times) / real_duration

synth_ground_truth = {}
with open('C:/BrainLink/data/nodeB_ground_truth.txt', 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        nid, ch, cell_type, rate, num_spikes = parts
        if int(ch) == 0 and cell_type == 'pyramidal':
            synth_rate = float(rate)
            break

print(f"\n=== METRIC 2: Firing Rate Comparison (pyramidal, ch1) ===")
print(f"Real unit_01 firing rate: {real_rate:.2f} Hz")
print(f"Synthetic pyramidal firing rate: {synth_rate:.2f} Hz")
print(f"Difference: {abs(real_rate - synth_rate):.2f} Hz")

# ============================================
# METRIC 3 - ISI DISTRIBUTION
# ============================================
real_isi = np.diff(real_times) * 1000  # ms

synth_isi = np.diff(np.array(crossings)) / SAMPLING_RATE * 1000  # ms

print(f"\n=== METRIC 3: Inter-Spike Interval ===")
print(f"Real mean ISI: {np.mean(real_isi):.2f} ms")
print(f"Synthetic mean ISI: {np.mean(synth_isi):.2f} ms")

ks_stat, ks_pvalue = stats.ks_2samp(real_isi, synth_isi)
print(f"KS test statistic: {ks_stat:.4f}, p-value: {ks_pvalue:.4f}")

# ============================================
# VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: waveform comparison
axes[0,0].plot(real_mean_waveform / np.max(np.abs(real_mean_waveform)),
                label='Real (Li et al. 2015)', linewidth=2)
if len(synth_waveforms) > 0 and len(synth_mean_waveform) == len(real_norm):
    axes[0,0].plot(synth_norm, label='Synthetic', linewidth=2, linestyle='--')
axes[0,0].set_title('Normalized Waveform Shape Comparison')
axes[0,0].set_xlabel('Samples')
axes[0,0].set_ylabel('Normalized Amplitude')
axes[0,0].legend()

# Panel 2: ISI histograms
axes[0,1].hist(real_isi, bins=50, alpha=0.5, label='Real', density=True, range=(0,2000))
axes[0,1].hist(synth_isi, bins=50, alpha=0.5, label='Synthetic', density=True, range=(0,2000))
axes[0,1].set_title('Inter-Spike Interval Distribution')
axes[0,1].set_xlabel('ISI (ms)')
axes[0,1].legend()

# Panel 3: firing rate comparison bar chart
axes[1,0].bar(['Real', 'Synthetic'], [real_rate, synth_rate], color=['steelblue', 'orange'])
axes[1,0].set_title('Firing Rate Comparison')
axes[1,0].set_ylabel('Firing Rate (Hz)')

# Panel 4: raw trace
axes[1,1].plot(synth_uv[:5000], linewidth=0.5)
axes[1,1].set_title('Synthetic Node B Trace (100ms)')
axes[1,1].set_xlabel('Samples')
axes[1,1].set_ylabel('Voltage (uV)')

plt.tight_layout()
plt.savefig('C:/BrainLink/data/nodeB_validation_figure.png', dpi=150)
plt.show()

# ============================================
# SAVE REPORT
# ============================================
with open('C:/BrainLink/data/nodeB_validation_report.txt', 'w') as f:
    f.write("=== BRAINLINK NODE B VALIDATION REPORT ===\n")
    f.write("Comparison: Synthetic motor cortex signal vs Real ALM-1\n\n")
    f.write(f"Real firing rate: {real_rate:.2f} Hz\n")
    f.write(f"Synthetic firing rate: {synth_rate:.2f} Hz\n\n")
    f.write(f"Real mean ISI: {np.mean(real_isi):.2f} ms\n")
    f.write(f"Synthetic mean ISI: {np.mean(synth_isi):.2f} ms\n")
    f.write(f"KS test p-value: {ks_pvalue:.4f}\n")

print("\nSaved: C:/BrainLink/data/nodeB_validation_figure.png")
print("Saved: C:/BrainLink/data/nodeB_validation_report.txt")
print("\n=== Node B Validation Complete ===")