import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile('C:/BrainLink/data/hc1/d13921/d1392101.dat', dtype='int16')
samples_per_channel = len(data) // 4
data = data[:samples_per_channel * 4].reshape(-1, 4)

# Convert to microvolts
data_uv = (data - 2048) * 0.5

# Parameters from previous run
noise = 31.3
threshold = -5 * noise

# Find spike crossings on channel 1
signal = data_uv[:, 0]
crossings = np.where(np.diff((signal < threshold).astype(int)) > 0)[0]

# Extract spike waveforms - 20 samples before, 40 after each spike
pre = 20
post = 40
waveforms = []

for idx in crossings:
    if idx > pre and idx + post < len(signal):
        waveform = signal[idx-pre:idx+post]
        waveforms.append(waveform)

waveforms = np.array(waveforms)
print('Total spikes extracted:', len(waveforms))
print('Waveform length:', waveforms.shape[1], 'samples')
print('Mean peak amplitude:', np.min(waveforms.mean(axis=0)), 'uV')
print('Spike duration at 20kHz:', 60/20000*1000, 'ms')

# Plot average waveform
plt.figure(figsize=(10,5))
plt.plot(waveforms[:100].T, color='blue', alpha=0.1)
plt.plot(waveforms.mean(axis=0), color='red', linewidth=2, label='Mean waveform')
plt.axvline(x=20, color='green', linestyle='--', label='Threshold crossing')
plt.title('Hippocampal Spike Waveforms - Channel 1')
plt.xlabel('Samples')
plt.ylabel('Voltage (uV)')
plt.legend()
plt.show()