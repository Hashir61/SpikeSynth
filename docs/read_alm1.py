"""
BrainLink - Motor Cortex Parameter Extraction
Source: CRCNS alm-1, Li et al. 2015, Nature 519:51-56
Animal: ANM218453, Session 20131015
Brain Region: Anterior Lateral Motor Cortex (ALM), Mouse
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

nwb_path = 'C:/BrainLink/data/data_structure_ANM218453_20131015.nwb'

print("=== BrainLink ALM-1 Parameter Extraction ===")
print(f"Source: Li et al. 2015, Nature, CRCNS alm-1")
print(f"Animal: ANM218453, Session: 20131015\n")

with h5py.File(nwb_path, 'r') as f:

    # ============================================
    # STEP 1 - GET VOLTAGE SCALING UNIT
    # ============================================
    ad_unit = f['general/extracellular_ephys/ADunit'][()]
    print(f"ADC unit: {ad_unit}")

    sampling_rate_str = f['general/extracellular_ephys/filtering'][()]
    print(f"Filtering info: {sampling_rate_str}")

    # ============================================
    # STEP 2 - GET CELL TYPES
    # ============================================
    cell_types_raw = f['processing/extracellular_units/UnitTimes/cell_types'][()]
    print(f"\n=== CELL TYPES ===")
    print(f"Raw cell types data: {cell_types_raw}")

    # ============================================
    # STEP 3 - EXTRACT SPIKE TIMES AND FIRING RATES
    # ============================================
    print(f"\n=== SPIKE STATISTICS PER UNIT ===")
    all_firing_rates = []
    all_isi_means = []
    unit_names = []

    event_waveform = f['processing/extracellular_units/EventWaveform']
    unit_times = f['processing/extracellular_units/UnitTimes']

    for unit_name in sorted(unit_times.keys()):
     if unit_name.startswith('unit_'):
        try:
            times = unit_times[f'{unit_name}/times'][()]
            if len(times) > 1:
                duration = times[-1] - times[0]
                if duration > 0:
                    rate = len(times) / duration
                    isi = np.diff(times) * 1000
                    all_firing_rates.append(rate)
                    all_isi_means.append(np.mean(isi))
                    unit_names.append(unit_name)
                    print(f"  {unit_name}: {len(times)} spikes, {rate:.2f} Hz, mean ISI: {np.mean(isi):.1f}ms")
        except Exception as e:
            print(f"  {unit_name}: skipped ({e})")
            if len(times) > 1:
                duration = times[-1] - times[0]
                if duration > 0:
                    rate = len(times) / duration
                    isi = np.diff(times) * 1000  # ms
                    all_firing_rates.append(rate)
                    all_isi_means.append(np.mean(isi))
                    unit_names.append(unit_name)
                    print(f"  {unit_name}: {len(times)} spikes, {rate:.2f} Hz, mean ISI: {np.mean(isi):.1f}ms")

    print(f"\n=== FIRING RATE SUMMARY ===")
    print(f"Units analyzed: {len(all_firing_rates)}")
    print(f"Mean firing rate: {np.mean(all_firing_rates):.2f} Hz")
    print(f"Min firing rate: {np.min(all_firing_rates):.2f} Hz")
    print(f"Max firing rate: {np.max(all_firing_rates):.2f} Hz")
    print(f"Mean ISI: {np.mean(all_isi_means):.2f} ms")

    # ============================================
    # STEP 4 - EXTRACT SPIKE WAVEFORMS
    # ============================================
    print(f"\n=== SPIKE WAVEFORMS ===")
    all_waveforms = []
    waveform_lengths = []

    for unit_name in sorted(event_waveform.keys()):
        if unit_name.startswith('unit_'):
            wf_data = event_waveform[f'{unit_name}/data'][()]
            sample_length = event_waveform[f'{unit_name}/sample_length'][()]
            print(f"  {unit_name}: waveform shape {wf_data.shape}, sample_length={sample_length}")
            if len(wf_data.shape) == 2:
                all_waveforms.append(wf_data)
                waveform_lengths.append(wf_data.shape[1])
            elif len(wf_data.shape) == 1:
                all_waveforms.append(wf_data.reshape(1, -1))
                waveform_lengths.append(len(wf_data))

    if all_waveforms:
        print(f"\nWaveform samples per spike: {waveform_lengths[0]}")

        # Stack all waveforms from all units
        all_wf_flat = np.vstack(all_waveforms)
        mean_waveform = all_wf_flat.mean(axis=0)
        peak_amplitude = np.min(mean_waveform)
        spike_duration_ms = waveform_lengths[0] / 20000 * 1000

        print(f"Total waveforms across all units: {len(all_wf_flat)}")
        print(f"Mean peak amplitude: {peak_amplitude:.4f} (raw ADC units)")
        print(f"Spike duration: {spike_duration_ms:.2f} ms")

    # ============================================
    # STEP 5 - GET ELECTRODE INFO
    # ============================================
    print(f"\n=== ELECTRODE INFORMATION ===")
    electrode_map = f['general/extracellular_ephys/electrode_map'][()]
    impedance = f['general/extracellular_ephys/impedance'][()]
    recording_type = f['general/extracellular_ephys/recording_type'][()]
    print(f"Recording type: {recording_type}")
    print(f"Electrode map shape: {electrode_map.shape}")
    print(f"Impedance: {impedance}")

    # ============================================
    # STEP 6 - SESSION INFO
    # ============================================
    session_desc = f['session_description'][()]
    institution = f['general/institution'][()]
    lab = f['general/lab'][()]
    print(f"\n=== SESSION INFO ===")
    print(f"Description: {session_desc}")
    print(f"Institution: {institution}")
    print(f"Lab: {lab}")

    # ============================================
    # STEP 7 - VISUALIZE WAVEFORMS
    # ============================================
    if all_waveforms:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Panel 1: individual unit waveforms
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_waveforms)))
        for i, (wf, name) in enumerate(zip(all_waveforms, unit_names)):
            mean_wf = wf.mean(axis=0) if len(wf.shape) == 2 else wf
            axes[0].plot(mean_wf, color=colors[i], label=name, linewidth=1.5)
        axes[0].set_title('ALM Motor Cortex - Mean Waveform Per Unit (Li et al. 2015)')
        axes[0].set_xlabel('Samples')
        axes[0].set_ylabel('Amplitude (raw ADC)')
        axes[0].legend(loc='right', fontsize=8)

        # Panel 2: firing rate distribution
        axes[1].bar(range(len(all_firing_rates)), all_firing_rates, color='steelblue')
        axes[1].set_title('Firing Rate Per Unit')
        axes[1].set_xlabel('Unit')
        axes[1].set_ylabel('Firing Rate (Hz)')
        axes[1].set_xticks(range(len(unit_names)))
        axes[1].set_xticklabels(unit_names, rotation=45, fontsize=8)

        plt.tight_layout()
        plt.savefig('C:/BrainLink/data/alm1_waveforms.png', dpi=150)
        plt.show()

print("\n=== ALM-1 Extraction Complete ===")