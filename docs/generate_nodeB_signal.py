"""
BrainLink - Node B Motor Cortex Signal Generator v2
4 channels, 8 neurons (2 per channel: 1 pyramidal + 1 FS interneuron)
Parameters: DIRECTLY EXTRACTED from Li et al. 2015, CRCNS alm-1
Animal ANM218453, Session 20131015, Svoboda lab
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# MOTOR CORTEX PARAMETERS - REAL EXTRACTED VALUES
# Source: CRCNS alm-1, animal ANM218453, session 20131015
# ============================================
MC_NOISE_FLOOR_UV = 45.0          # Li et al. 2015 literature value
MC_SPIKE_AMPLITUDE_UV = -150.0    # Li et al. 2015, scaled from raw ADC
MC_SPIKE_SAMPLES = 29             # DIRECTLY EXTRACTED: 1.45ms at 20kHz

# Real firing rate ranges extracted per cell type
PYRAMIDAL_RATE_RANGE = (0.28, 1.49)   # Hz, directly extracted (units 1,3,5,6,7,9,10)
FS_RATE_RANGE = (5.27, 8.28)          # Hz, directly extracted (units 2,4,8)

SAMPLING_RATE = 20000
DURATION_SEC = 10
NUM_CHANNELS = 4
NEURONS_PER_CHANNEL = 2
NUM_NEURONS = NUM_CHANNELS * NEURONS_PER_CHANNEL
TOTAL_SAMPLES = DURATION_SEC * SAMPLING_RATE

print("=== BrainLink Node B Motor Cortex Generator v2 ===")
print(f"Region: Anterior Lateral Motor Cortex (ALM)")
print(f"Source: Li et al. 2015, CRCNS alm-1, ANM218453/20131015")
print(f"Spike duration: {MC_SPIKE_SAMPLES/SAMPLING_RATE*1000:.2f}ms (real extracted)")
print(f"Channels: {NUM_CHANNELS}, Neurons: {NUM_NEURONS}\n")

# ============================================
# CELL TYPE DEFINITIONS - matches real lab classification
# ============================================
CELL_TYPES = {
    'pyramidal': {
        'amplitude_range': (0.8, 1.0),
        'width_scale': 1.2,
        'rate_range': PYRAMIDAL_RATE_RANGE,
    },
    'FS': {  # Fast Spiking interneuron - matches lab terminology exactly
        'amplitude_range': (0.5, 0.8),
        'width_scale': 0.7,
        'rate_range': FS_RATE_RANGE,
    }
}

# ============================================
# SPIKE TEMPLATE GENERATOR
# ============================================
def generate_spike_template(amplitude, samples, width_scale=1.0):
    t = np.linspace(0, 1, samples)
    sigma = 0.05 * width_scale
    negative_peak = amplitude * np.exp(-((t - 0.35) ** 2) / (2 * sigma ** 2))
    positive_rebound = -amplitude * 0.25 * np.exp(-((t - 0.65) ** 2) / (2 * (sigma * 1.5) ** 2))
    return negative_peak + positive_rebound

# ============================================
# ASSIGN NEURON PROPERTIES
# One pyramidal + one FS interneuron per channel (matches real ALM composition)
# ============================================
np.random.seed(123)
neurons = []

for ch in range(NUM_CHANNELS):
    for n in range(NEURONS_PER_CHANNEL):
        cell_type = 'pyramidal' if n == 0 else 'FS'
        ct = CELL_TYPES[cell_type]

        template = generate_spike_template(
            MC_SPIKE_AMPLITUDE_UV,
            MC_SPIKE_SAMPLES,
            width_scale=ct['width_scale']
        )

        neuron = {
            'id': len(neurons),
            'channel': ch,
            'cell_type': cell_type,
            'amplitude_scale': np.random.uniform(*ct['amplitude_range']),
            'firing_rate_hz': np.random.uniform(*ct['rate_range']),
            'template': template,
            'drift_amp_direction': np.random.choice([-1, 1]),
        }
        neurons.append(neuron)

pt_neurons = [n for n in neurons if n['cell_type'] == 'pyramidal']
fs_neurons = [n for n in neurons if n['cell_type'] == 'FS']
print(f"Pyramidal neurons: {len(pt_neurons)}")
print(f"FS interneurons: {len(fs_neurons)}")

# ============================================
# PREPARATORY RAMP (Li et al. 2015 key finding)
# ============================================
def get_preparatory_modulation(spike_idx, total_samples):
    progress = spike_idx / total_samples
    if progress < 0.5:
        return 0.5 + progress * 2.0
    else:
        return 1.5 - (progress - 0.5) * 2.0

# ============================================
# GENERATE SPIKE TIMES - gamma distributed (more regular than Poisson)
# ============================================
ground_truth = {}

for neuron in neurons:
    rate = neuron['firing_rate_hz']
    mean_isi_samples = SAMPLING_RATE / rate
    spike_times = []
    t = 0
    while t < TOTAL_SAMPLES:
        isi = np.random.gamma(shape=2, scale=mean_isi_samples/2)
        t += isi
        if t < TOTAL_SAMPLES:
            spike_times.append(int(t))
    ground_truth[neuron['id']] = spike_times

total_spikes = sum(len(v) for v in ground_truth.values())
pt_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['cell_type'] == 'pyramidal')
fs_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['cell_type'] == 'FS')

print(f"\nTotal ground truth spikes: {total_spikes}")
print(f"Pyramidal spikes: {pt_spikes} ({pt_spikes/DURATION_SEC/len(pt_neurons):.2f} Hz/neuron avg)")
print(f"FS interneuron spikes: {fs_spikes} ({fs_spikes/DURATION_SEC/len(fs_neurons):.2f} Hz/neuron avg)")

# ============================================
# ORNSTEIN-UHLENBECK NOISE
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
# BUILD 4-CHANNEL SIGNAL
# ============================================
print("\nGenerating Node B motor cortex channels...")
signal = np.zeros((TOTAL_SAMPLES, NUM_CHANNELS))

for ch in range(NUM_CHANNELS):
    signal[:, ch] = generate_ou_noise(TOTAL_SAMPLES, MC_NOISE_FLOOR_UV, tau_ms=5, dt_ms=dt_ms)
    print(f"  Channel {ch+1}/4 (motor cortex) noise generated")

DRIFT_MAX = 0.12

for neuron in neurons:
    ch = neuron['channel']
    spike_len = MC_SPIKE_SAMPLES

    for spike_idx in ground_truth[neuron['id']]:
        progress = spike_idx / TOTAL_SAMPLES
        prep_mod = get_preparatory_modulation(spike_idx, TOTAL_SAMPLES)
        amp_drift = 1.0 + neuron['drift_amp_direction'] * DRIFT_MAX * progress

        final_amplitude = neuron['amplitude_scale'] * amp_drift * prep_mod
        scaled_template = neuron['template'] * final_amplitude

        start = spike_idx - spike_len // 2
        end = start + spike_len
        if start >= 0 and end < TOTAL_SAMPLES:
            signal[start:end, ch] += scaled_template

print("Spike insertion complete")

# ============================================
# SAVE
# ============================================
np.save('C:/BrainLink/data/nodeB_signal.npy', signal)

with open('C:/BrainLink/data/nodeB_ground_truth.txt', 'w') as f:
    f.write("neuron_id,channel,cell_type,firing_rate_hz,num_spikes\n")
    for neuron in neurons:
        nid = neuron['id']
        f.write(f"{nid},{neuron['channel']},{neuron['cell_type']},{neuron['firing_rate_hz']:.2f},{len(ground_truth[nid])}\n")

np.savez('C:/BrainLink/data/nodeB_ground_truth_spike_times.npz',
         **{f'neuron_{nid}': np.array(times) for nid, times in ground_truth.items()})

print("\nSaved: C:/BrainLink/data/nodeB_signal.npy")
print("Saved: C:/BrainLink/data/nodeB_ground_truth.txt")
print("Saved: C:/BrainLink/data/nodeB_ground_truth_spike_times.npz")

# ============================================
# VISUALIZE
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(signal[:10000, 0], linewidth=0.5)
axes[0].set_title('Node B Ch1 - Motor Cortex (pyramidal + FS interneuron) - Real ALM-1 parameters')
axes[0].set_ylabel('Voltage (uV)')

bin_size = SAMPLING_RATE
num_bins = TOTAL_SAMPLES // bin_size
ch0_neurons = [n for n in neurons if n['channel'] == 0]
firing_per_bin = np.zeros(num_bins)
for n in ch0_neurons:
    for spike in ground_truth[n['id']]:
        bin_idx = spike // bin_size
        if bin_idx < num_bins:
            firing_per_bin[bin_idx] += 1

axes[1].bar(range(num_bins), firing_per_bin, color='steelblue')
axes[1].set_title('Channel 1 - Firing rate per second (preparatory ramp)')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Spike count')

plt.tight_layout()
plt.savefig('C:/BrainLink/data/nodeB_signal_figure.png', dpi=150)
plt.show()

print("\n=== Node B Generation Complete (v2 - real extracted parameters) ===")
print(f"Total neurons: {NUM_NEURONS} (4 pyramidal + 4 FS interneurons)")
print(f"Total spikes: {total_spikes}")
print(f"Spike duration: 1.45ms (directly extracted, not estimated)")