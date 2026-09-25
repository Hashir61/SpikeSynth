"""
SpikeSynth — Open-source synthetic neural signal generation library
Generates electrophysiologically realistic multi-region neural signals
parameterized from peer-reviewed experimental datasets.

Supported regions:
  - Hippocampus CA1 (Henze et al. 2000, CRCNS hc-1)
  - Prefrontal Cortex (Kepecs lab, CRCNS pfc-2)
  - Motor Cortex ALM (Li et al. 2015, CRCNS alm-1)

Usage:
  from spikesynth.generator import generate_signal
  from spikesynth.validator import validate_signal
"""

__version__ = "0.1.0"
__author__ = "Hashir Usman"
__license__ = "MIT"