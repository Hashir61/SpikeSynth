import numpy as np

print("=== Loading Ground Truth ===")

ground_truth = {}
with open('C:/BrainLink/data/ground_truth.txt', 'r') as f:
    next(f)
    for line in f:
        parts = line.strip().split(',')
        nid, ch, region, rate, num_spikes = parts
        ground_truth[int(nid)] = {
            'channel': int(ch),
            'region': region,
            'firing_rate': float(rate),
            'num_spikes': int(num_spikes)
        }

print(f"Loaded {len(ground_truth)} neurons")

hc = [n for n in ground_truth.values() if n['region'] == 'hippocampus']
pfc = [n for n in ground_truth.values() if n['region'] == 'pfc']

print(f"Hippocampal neurons: {len(hc)}")
print(f"PFC neurons: {len(pfc)}")
print(f"Total ground truth spikes: {sum(n['num_spikes'] for n in ground_truth.values())}")
print(f"HC total spikes: {sum(n['num_spikes'] for n in hc)}")
print(f"PFC total spikes: {sum(n['num_spikes'] for n in pfc)}")
print("\nFramework ready for STM32 sorter output.")