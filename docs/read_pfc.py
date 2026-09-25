"""
BrainLink - PFC Spike Waveform Extraction and Visualization
Source: CRCNS pfc-2, Kepecs lab, session EE0622/EE.043
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

SAMPLING_RATE = 20000
WAVEFORM_SAMPLES = 32

spk_path = 'C:/BrainLink/data/pfc1/EE.043/EE.043.spk.1'

print("=== Loading PFC spike waveforms ===")
print(f"Reading: {spk_path}")

if not os.path.exists(spk_path):
    print("ERROR: file not found at that path")
else:
    spk_data = np.fromfile(spk_path, dtype='int16')
    print(f"Raw samples loaded: {len(spk_data)}")

    num_waveforms = len(spk_data) // WAVEFORM_SAMPLES
    waveforms = spk_data[:num_waveforms * WAVEFORM_SAMPLES].reshape(-1, WAVEFORM_SAMPLES)
    waveforms_uv = waveforms.astype(float) * 0.195  # pfc-2 standard scaling factor

    mean_waveform = waveforms_uv.mean(axis=0)
    peak_amplitude = np.min(mean_waveform)

    print(f"Total waveforms: {len(waveforms_uv)}")
    print(f"Waveform samples: {WAVEFORM_SAMPLES}")
    print(f"Mean peak amplitude: {peak_amplitude:.2f} uV")

    fig = plt.figure(figsize=(10, 6))
    for i in range(min(100, len(waveforms_uv))):
        plt.plot(waveforms_uv[i], color='blue', alpha=0.1)
    plt.plot(mean_waveform, color='red', linewidth=2, label='Mean waveform')
    plt.title('PFC Spike Waveforms - Shank 1 (Kepecs lab, CRCNS pfc-2)')
    plt.xlabel('Samples')
    plt.ylabel('Voltage (uV)')
    plt.legend()
    plt.tight_layout()

    save_path = 'C:/BrainLink/data/pfc_waveforms.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved figure to: {save_path}")

    plt.show()

    print("=== Done ===")