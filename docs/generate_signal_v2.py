"""
BrainLink - Signal Generator v2
Adds three realism upgrades over v1:
1. Burst firing for hippocampal place cells (real place cells fire in rapid bursts, not pure Poisson)
2. Waveform drift over time (electrode-tissue interface shifts slowly, changing spike shape)
3. Spike collisions (two neurons firing near-simultaneously on the same channel, merging waveforms)

These are documented failure modes / realistic complications in published spike sorting
literature (Harris et al. 2000; Pillow et al. 2013 on burst statistics;
Quiroga et al. 2004 on collision handling). Adding them makes the sorting problem
non-trivial, which is what justifies a two-chip pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PARAMETERS (same as v1)
# ============================================
HC_NOISE_FLOOR_UV = 31.3
HC_SPIKE_AMPLITUDE_UV = -179.78
HC_FIRING_RATE_RANGE = (1, 40)
HC_SPIKE_SAMPLES = 60

PFC_NOISE_FLOOR_UV = 35.0
PFC_SPIKE_AMPLITUDE_UV = -110.0
PFC_FIRING_RATE_RANGE = (1, 5)
PFC_SPIKE_SAMPLES = 32

SAMPLING_RATE = 20000
DURATION_SEC = 10
NUM_CHANNELS = 8
NEURONS_PER_CHANNEL = 3
NUM_NEURONS = NUM_CHANNELS * NEURONS_PER_CHANNEL
TOTAL_SAMPLES = DURATION_SEC * SAMPLING_RATE

# ============================================
# NEW PARAMETER - BURST FIRING (hippocampus only)
# Place cells fire in theta-bursts: 2-6 spikes packed within
# ~20-40ms, riding on top of the slower Poisson envelope.
# This matches O'Keefe & Recce 1993 "theta phase precession" literature.
# ============================================
BURST_PROBABILITY = 0.35       # fraction of HC spike events that become a burst
BURST_SIZE_RANGE = (2, 5)      # spikes per burst
BURST_ISI_MS = (3, 8)          # tight spacing within a burst

# ============================================
# NEW PARAMETER - WAVEFORM DRIFT
# Electrode-tissue interface shifts slowly (micromotion, glial encapsulation)
# causing spike amplitude to drift +/-15% over the recording,
# and slight shape broadening over time.
# Cited: Dickey et al. 2009 on electrode drift in chronic recordings.
# ============================================
DRIFT_MAX_AMPLITUDE_PCT = 0.15  # max 15% amplitude drift over full recording
DRIFT_MAX_WIDTH_PCT = 0.10      # max 10% width change over full recording

print("=== BrainLink Generator v2 (Realism Upgrades) ===")
print(f"Burst firing: enabled (hippocampus, p={BURST_PROBABILITY})")
print(f"Waveform drift: enabled (+/-{DRIFT_MAX_AMPLITUDE_PCT*100:.0f}% amplitude)")
print(f"Spike collisions: enabled (emergent from overlapping independent spike trains)\n")

# ============================================
# SPIKE TEMPLATE GENERATOR
# ============================================
def generate_spike_template(amplitude, samples):
    t = np.linspace(0, 1, samples)
    negative_peak = amplitude * np.exp(-((t - 0.35) ** 2) / (2 * 0.05 ** 2))
    positive_rebound = -amplitude * 0.3 * np.exp(-((t - 0.6) ** 2) / (2 * 0.08 ** 2))
    return negative_peak + positive_rebound

hc_template = generate_spike_template(HC_SPIKE_AMPLITUDE_UV, HC_SPIKE_SAMPLES)
pfc_template = generate_spike_template(PFC_SPIKE_AMPLITUDE_UV, PFC_SPIKE_SAMPLES)

def stretch_waveform(template, width_scale):
    """Resample a waveform to simulate width change (drift) using linear interpolation."""
    original_len = len(template)
    new_len = max(int(original_len * width_scale), 5)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    stretched = np.interp(x_new, x_old, template)
    # resample back to original_len so it still fits the same array slot
    x_final = np.linspace(0, 1, original_len)
    x_stretched_axis = np.linspace(0, 1, new_len)
    return np.interp(x_final, x_stretched_axis, stretched)

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
            'spike_len': spike_len,
            # per-neuron random drift direction, so not all neurons drift the same way
            'drift_amp_direction': np.random.choice([-1, 1]),
            'drift_width_direction': np.random.choice([-1, 1]),
        }
        neurons.append(neuron)

hc_count = sum(1 for n in neurons if n['region'] == 'hippocampus')
pfc_count = sum(1 for n in neurons if n['region'] == 'pfc')
print(f"Hippocampal neurons: {hc_count}, PFC neurons: {pfc_count}")

# ============================================
# GENERATE SPIKE TIMES - WITH BURST FIRING FOR HIPPOCAMPUS
# ============================================
def generate_spike_train_with_bursts(rate_hz, total_samples, region):
    """
    For hippocampus: each Poisson 'event' has a chance to expand into a burst
    of 2-5 closely spaced spikes (theta-burst firing).
    For PFC: pure Poisson, no bursting (matches sparse working-memory firing).
    """
    mean_isi_samples = SAMPLING_RATE / rate_hz
    spike_times = []
    t = 0
    while t < total_samples:
        isi = np.random.exponential(mean_isi_samples)
        t += isi
        if t >= total_samples:
            break

        if region == 'hippocampus' and np.random.random() < BURST_PROBABILITY:
            # expand this event into a burst
            burst_size = np.random.randint(*BURST_SIZE_RANGE)
            burst_t = t
            for _ in range(burst_size):
                if burst_t < total_samples:
                    spike_times.append(int(burst_t))
                burst_isi_ms = np.random.uniform(*BURST_ISI_MS)
                burst_t += burst_isi_ms / 1000 * SAMPLING_RATE
            t = burst_t  # continue main ISI process after the burst
        else:
            spike_times.append(int(t))

    return spike_times

ground_truth = {}
for neuron in neurons:
    ground_truth[neuron['id']] = generate_spike_train_with_bursts(
        neuron['firing_rate_hz'], TOTAL_SAMPLES, neuron['region']
    )

total_spikes = sum(len(v) for v in ground_truth.values())
hc_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['region'] == 'hippocampus')
pfc_spikes = sum(len(ground_truth[n['id']]) for n in neurons if n['region'] == 'pfc')

print(f"\nTotal ground truth spikes: {total_spikes}")
print(f"Hippocampal spikes: {hc_spikes} (includes burst-expanded events)")
print(f"PFC spikes: {pfc_spikes}")

# ============================================
# DETECT COLLISIONS (for reporting purposes)
# A collision = two different neurons on the SAME channel firing
# within one spike-window of each other. This happens naturally
# once bursts + multiple neurons per channel are combined - we don't
# force it, we just measure how often it occurs.
# ============================================
collision_count = 0
for ch in range(NUM_CHANNELS):
    ch_neurons = [n for n in neurons if n['channel'] == ch]
    spike_len = ch_neurons[0]['spike_len']
    all_times = []
    for n in ch_neurons:
        all_times.extend(ground_truth[n['id']])
    all_times = np.sort(np.array(all_times))
    if len(all_times) > 1:
        gaps = np.diff(all_times)
        collision_count += np.sum(gaps < spike_len)

print(f"Emergent spike collisions detected: {collision_count}")
print("(Two neurons on the same channel firing close enough to overlap waveforms)")

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
# BUILD FULL 8-CHANNEL SIGNAL WITH DRIFT
# ============================================
print("\nGenerating channels with drift + noise...")
signal = np.zeros((TOTAL_SAMPLES, NUM_CHANNELS))

for ch in range(NUM_CHANNELS):
    region = 'hippocampus' if ch < 4 else 'pfc'
    noise_level = HC_NOISE_FLOOR_UV if region == 'hippocampus' else PFC_NOISE_FLOOR_UV
    signal[:, ch] = generate_ou_noise(TOTAL_SAMPLES, noise_level, tau_ms=5, dt_ms=dt_ms)
    print(f"  Channel {ch+1}/8 ({region}) noise generated")

for neuron in neurons:
    ch = neuron['channel']
    base_template = neuron['template']
    spike_len = neuron['spike_len']
    spikes = ground_truth[neuron['id']]

    for spike_idx in spikes:
        # progress through the recording, 0.0 at start, 1.0 at end
        progress = spike_idx / TOTAL_SAMPLES

        # amplitude drift: scale changes linearly across the recording
        amp_drift = 1.0 + neuron['drift_amp_direction'] * DRIFT_MAX_AMPLITUDE_PCT * progress
        width_drift = 1.0 + neuron['drift_width_direction'] * DRIFT_MAX_WIDTH_PCT * progress

        drifted_template = stretch_waveform(base_template, width_drift) * neuron['amplitude_scale'] * amp_drift

        start = spike_idx - spike_len // 2
        end = start + spike_len
        if start >= 0 and end < TOTAL_SAMPLES:
            signal[start:end, ch] += drifted_template

print("Spike insertion with drift complete")

# ============================================
# SAVE SIGNAL, GROUND TRUTH, AND PER-NEURON SPIKE TIMES
# (v1 only saved counts - v2 saves actual spike sample indices,
#  needed for accuracy_framework.py to do real tolerance-based matching)
# ============================================
np.save('C:/BrainLink/data/synthetic_signal_v2.npy', signal)

with open('C:/BrainLink/data/ground_truth_v2.txt', 'w') as f:
    f.write("neuron_id,channel,region,firing_rate_hz,num_spikes\n")
    for neuron in neurons:
        nid = neuron['id']
        f.write(f"{nid},{neuron['channel']},{neuron['region']},{neuron['firing_rate_hz']:.2f},{len(ground_truth[nid])}\n")

# Save exact spike times per neuron - needed for real accuracy testing later
np.savez('C:/BrainLink/data/ground_truth_spike_times.npz',
         **{f'neuron_{nid}': np.array(times) for nid, times in ground_truth.items()})

print("\nSaved: C:/BrainLink/data/synthetic_signal_v2.npy")
print("Saved: C:/BrainLink/data/ground_truth_v2.txt")
print("Saved: C:/BrainLink/data/ground_truth_spike_times.npz (exact spike sample indices)")

# ============================================
# VISUALIZE - SHOW BURST + DRIFT EFFECTS
# ============================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Panel 1: hippocampal channel showing bursts
axes[0].plot(signal[:20000, 0], linewidth=0.5)
axes[0].set_title('Hippocampus Ch1 - Note tight burst clusters (theta-burst firing)')
axes[0].set_ylabel('Voltage (uV)')

# Panel 2: same neuron's waveform at start vs end of recording (drift proof)
neuron0_spikes = ground_truth[0]
if len(neuron0_spikes) > 2:
    early_spike = neuron0_spikes[0]
    late_spike = neuron0_spikes[-1]
    spike_len = neurons[0]['spike_len']
    if early_spike + spike_len < TOTAL_SAMPLES and late_spike + spike_len < TOTAL_SAMPLES:
        early_wave = signal[early_spike - spike_len//2 : early_spike + spike_len//2, 0]
        late_wave = signal[late_spike - spike_len//2 : late_spike + spike_len//2, 0]
        axes[1].plot(early_wave, label='Early in recording', linewidth=1.5)
        axes[1].plot(late_wave, label='Late in recording', linewidth=1.5)
        axes[1].set_title('Neuron 0 waveform: early vs late (drift visible)')
        axes[1].set_ylabel('Voltage (uV)')
        axes[1].legend()

# Panel 3: zoomed view showing a collision if one exists
axes[2].plot(signal[:5000, 1], linewidth=0.7)
axes[2].set_title('Hippocampus Ch2 - Zoomed view (look for overlapping/merged spikes)')
axes[2].set_ylabel('Voltage (uV)')
axes[2].set_xlabel('Samples')

plt.tight_layout()
plt.savefig('C:/BrainLink/data/v2_realism_figure.png', dpi=150)
plt.show()

print("\n=== v2 Generation Complete ===")
print(f"Total neurons: {NUM_NEURONS}")
print(f"Total spikes: {total_spikes}")
print(f"Emergent collisions: {collision_count}")
print(f"Burst events included: yes (hippocampus)")
print(f"Waveform drift included: yes (+/-{DRIFT_MAX_AMPLITUDE_PCT*100:.0f}% amplitude)")