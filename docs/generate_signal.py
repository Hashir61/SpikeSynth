"""
BrainLink - Unified 8-Channel Synthetic Neural Signal Generator
Node A: Hippocampus (ch 1-4) + Prefrontal Cortex (ch 5-8)
Parameters from Henze et al. 2000 (hippocampus) and Kepecs lab pfc-2 (PFC)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# HIPPOCAMPUS PARAMETERS (Channels 1-4)
# Source: Henze et al. 2000, CRCNS hc-1, d13921
# ============================================
HC_NOISE_FLOOR_UV = 31.3
HC_SPIKE_AMPLITUDE_UV = -179.78
HC_FIRING_RATE_RANGE = (1, 40)  # Hz, per neuron, Softky & Koch 1993
HC_SPIKE_SAMPLES = 60           # 3.0ms at 20kHz

# ============================================
# PFC PARAMETERS (Channels 5-8)
# Source: Kepecs lab pfc-2, EE.043 + Fujisawa 2008 + Benchenane 2010
# ============================================
PFC_NOISE_FLOOR_UV = 35.0
PFC_SPIKE_AMPLITUDE_UV = -110.0
PFC_FIRING_RATE_RANGE = (1, 5)   # Hz, sparse PFC firing
PFC_SPIKE_SAMPLES = 32           # 1.6ms at 20kHz

SAMPLING_RATE = 20000
DURATION_SEC = 10
NUM_CHANNELS = 8
NEURONS_PER_CHANNEL = 3
NUM_NEURONS = NUM_CHANNELS * NEURONS_PER_CHANNEL  # 24 total
TOTAL_SAMPLES = DURATION_SEC * SAMPLING_RATE

print("=== BrainLink Unified 8-Channel Generator ===")
print(f"Channels 1-4: Hippocampus CA1")
print(f"Channels 5-8: Prefrontal Cortex")
print(f"Total neurons: {NUM_NEURONS}")
print(f"Duration: {DURATION_SEC}s\n")

# ============================================
# SPIKE TEMPLATE GENERATOR
# Different shape per region based on real waveform duration
# ============================================
def generate_spike_template(amplitude, samples):
    t = np.linspace(0, 1, samples)
    negative_peak = amplitude * np.exp(-((t - 0.35) ** 2) / (2 * 0.05 ** 2))
    positive_rebound = -amplitude * 0.3 * np.exp(-((t - 0.6) ** 2) / (2 * 0.08 ** 2))
    return negative_peak + positive_rebound

hc_template = generate_spike_template(HC_SPIKE_AMPLITUDE_UV, HC_SPIKE_SAMPLES)
pfc_template = generate_spike_template(PFC_SPIKE_AMPLITUDE_UV, PFC_SPIKE_SAMPLES)

# ============================================
# ASSIGN NEURON PROPERTIES BY REGION
# ============================================
np.random.seed(42)
neurons = []

for ch in range(NUM_CHANNELS):
    region = 'hippocampus' if ch < 4 else 'pfc'
    rate_range = HC_FIRING_RATE_RANGE if region == 'hippocampus' else PFC_FIRING_RATE_RANGE
    template = hc_template if region == 'hippocampus' else pfc_template
    spike_len = HC_SPIKE_SAMPLES if region == 'hippocampus' else PFC_SPIKE_SAMPLES

    for n in range(NEURONS_PER_CHANNEL):
        neuron = {
            'id': len(neurons),
            'channel': ch,
            'region': region,
            'amplitude_scale': np.random.uniform(0.7, 1.3),
            'firing_rate_hz': np.random.uniform(*rate_range),
            'template': template,
            'spike_len': spike_len
        }
        neurons.append(neuron)

hc_count = sum(1 for n in neurons if n['region'] == 'hippocampus')
pfc_count = sum(1 for n in neurons if n['region'] == 'pfc')
print(f"Hippocampal neurons: {hc_count}")
print(f"PFC neurons: {pfc_count}")

# ============================================
# GENERATE POISSON SPIKE TIMES
# ============================================
ground_truth = {}

for neuron in neurons:
    rate = neuron['firing_rate_hz']
    mean_isi_samples = SAMPLING_RATE / rate
    spike_times = []
    t = 0
    while t < TOTAL_SAMPLES:
        isi = np.random.exponential(mean_isi_samples)
        t += isi
        if t < TOTAL_SAMPLES:
            spike_times.append(int(t))
    ground_truth[neuron['id']] = spike_times

total_spikes = sum(len(v) for v in ground_truth.values())
print(f"\nTotal ground truth spikes: {total_spikes}")

hc_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['region'] == 'hippocampus')
pfc_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['region'] == 'pfc')
print(f"Hippocampal spikes: {hc_spikes} ({hc_spikes/DURATION_SEC/hc_count:.2f} Hz/neuron avg)")
print(f"PFC spikes: {pfc_spikes} ({pfc_spikes/DURATION_SEC/pfc_count:.2f} Hz/neuron avg)")

# ============================================
# ORNSTEIN-UHLENBECK NOISE GENERATOR
# ============================================
def generate_ou_noise(num_samples, sigma, tau_ms, dt_ms):
    tau_samples = tau_ms / dt_ms
    noise = np.zeros(num_samples)
    for i in range(1, num_samples):
        dW = np.random.normal(0, 1)
        noise[i] = noise[i-1] - (noise[i-1] / tau_samples) + sigma * np.sqrt(2/tau_samples) * dW
    return noise

dt_ms = 1000 / SAMPLING_RATE

# ============================================
# BUILD FULL 8-CHANNEL SIGNAL
# ============================================
print("\nGenerating channels...")
signal = np.zeros((TOTAL_SAMPLES, NUM_CHANNELS))

for ch in range(NUM_CHANNELS):
    region = 'hippocampus' if ch < 4 else 'pfc'
    noise_level = HC_NOISE_FLOOR_UV if region == 'hippocampus' else PFC_NOISE_FLOOR_UV
    channel_noise = generate_ou_noise(TOTAL_SAMPLES, noise_level, tau_ms=5, dt_ms=dt_ms)
    signal[:, ch] = channel_noise
    print(f"  Channel {ch+1}/8 ({region}) noise generated")

for neuron in neurons:
    ch = neuron['channel']
    scaled_template = neuron['template'] * neuron['amplitude_scale']
    spike_len = neuron['spike_len']

    for spike_idx in ground_truth[neuron['id']]:
        start = spike_idx - spike_len // 2
        end = start + spike_len
        if start >= 0 and end < TOTAL_SAMPLES:
            signal[start:end, ch] += scaled_template

print("Spike insertion complete")

# ============================================
# SAVE SIGNAL AND GROUND TRUTH
# ============================================
np.save('C:/BrainLink/data/synthetic_signal.npy', signal)

with open('C:/BrainLink/data/ground_truth.txt', 'w') as f:
    f.write("neuron_id,channel,region,firing_rate_hz,num_spikes\n")
    for neuron in neurons:
        nid = neuron['id']
        spikes = ground_truth[nid]
        f.write(f"{nid},{neuron['channel']},{neuron['region']},{neuron['firing_rate_hz']:.2f},{len(spikes)}\n")

print("\nSaved: C:/BrainLink/data/synthetic_signal.npy")
print("Saved: C:/BrainLink/data/ground_truth.txt")

# ============================================
# VISUALIZE - SHOW BOTH REGIONS
# ============================================
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
show_channels = [0, 1, 4, 5]  # 2 hippocampal, 2 PFC
labels = ['Hippocampus Ch1', 'Hippocampus Ch2', 'PFC Ch5', 'PFC Ch6']

for i, ch in enumerate(show_channels):
    axes[i].plot(signal[:20000, ch], linewidth=0.5)
    axes[i].set_title(f'{labels[i]} - Synthetic Signal (first 1 second)')
    axes[i].set_ylabel('Voltage (uV)')
axes[-1].set_xlabel('Samples')
plt.tight_layout()
plt.show()

print("\n=== Generation Complete ===")
print(f"Total neurons: {NUM_NEURONS} ({hc_count} hippocampal, {pfc_count} PFC)")
print(f"Total spikes: {total_spikes}")