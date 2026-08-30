import json, itertools, numpy as np
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
levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / np.sqrt(10.0)
rms = np.sqrt(np.mean(np.abs(sy) ** 2))
normalized = sy / rms
bitpairs = [(0, 0), (0, 1), (1, 1), (1, 0)]

best = []
for swap_axes in [False, True]:
    for order in ['RI', 'IR']:
        for real_perm in itertools.permutations(range(4)):
            real_map = {idx: bitpairs[real_perm[idx]] for idx in range(4)}
            for imag_perm in itertools.permutations(range(4)):
                imag_map = {idx: bitpairs[imag_perm[idx]] for idx in range(4)}
                bits = []
                for s in normalized:
                    ri = int(np.argmin(np.abs(s.real - levels)))
                    ii = int(np.argmin(np.abs(s.imag - levels)))
                    if swap_axes:
                        ri, ii = ii, ri
                    if order == 'RI':
                        bits.extend(real_map[ri])
                        bits.extend(imag_map[ii])
                    else:
                        bits.extend(imag_map[ii])
                        bits.extend(real_map[ri])
                bits = np.asarray(bits, dtype=np.uint8)
                acc = np.mean(bits[:len(exp)] == exp) * 100
                if acc > 90:
                    best.append((acc, swap_axes, order, real_perm, imag_perm))

print('count', len(best))
for item in sorted(best, reverse=True)[:20]:
    print(item)
