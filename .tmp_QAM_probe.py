import json, numpy as np, itertools
from pathlib import Path
from ai.preprocessing.iq_loader import load_iq_file
from signal_processing.synchronization.frequency_sync import correct_frequency_offset
from signal_processing.synchronization.phase_sync import correct_phase_offset
from signal_processing.synchronization.timing_sync import estimate_timing_offset, sample_symbols

meta = json.loads(Path('data/metadata/QAM16/signal_0601_16qam.json').read_text())
iq = load_iq_file('data/iq/QAM16/signal_0601_qam16.iq')
fc = correct_frequency_offset(iq, meta['frequency_offset_hz'], meta['sampling_frequency_hz'])
pc = correct_phase_offset(fc, meta['phase_offset_degrees'])
off = estimate_timing_offset(pc, meta['samples_per_symbol'])
sy = sample_symbols(pc, meta['samples_per_symbol'], off)
exp = np.array([int(b) for b in meta['bits']], dtype=np.uint8)
levels = np.array([-3, -1, 1, 3], dtype=np.float64) / np.sqrt(10)
rms = np.sqrt(np.mean(np.abs(sy) ** 2))
normalized = sy / rms
bitpairs = [(0, 0), (0, 1), (1, 1), (1, 0)]

def decode_with(angle_deg, swap, flip_r, flip_i, real_order, imag_order):
    angle = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    real_map = {i: bitpairs[real_order[i]] for i in range(4)}
    imag_map = {i: bitpairs[imag_order[i]] for i in range(4)}
    bits = []
    for s in normalized:
        v = np.array([s.real, s.imag], dtype=np.float64)
        v = R @ v
        if swap:
            v = v[[1, 0]]
        v[0] *= flip_r
        v[1] *= flip_i
        ri = int(np.argmin(np.abs(v[0] - levels)))
        ii = int(np.argmin(np.abs(v[1] - levels)))
        bits.extend(real_map[ri])
        bits.extend(imag_map[ii])
    bits = np.asarray(bits, dtype=np.uint8)
    return np.mean(bits[:len(exp)] == exp) * 100

best = []
angles = [0, 45, 90, 135, 180, 225, 270, 315]
for angle in angles:
    for swap in [False, True]:
        for flip_r in [1, -1]:
            for flip_i in [1, -1]:
                for real_order in itertools.permutations(range(4)):
                    for imag_order in itertools.permutations(range(4)):
                        acc = decode_with(angle, swap, flip_r, flip_i, real_order, imag_order)
                        if acc >= 95:
                            best.append((acc, angle, swap, flip_r, flip_i, real_order, imag_order))
print('BEST COUNT', len(best))
for item in sorted(best, reverse=True)[:20]:
    print(item)
