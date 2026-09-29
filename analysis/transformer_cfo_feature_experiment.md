# Transformer Local Circular Differential Phase Feature Experiment

## 1. Overview & Motivation

Following the error analysis of the baseline Transformer model (which achieved **87.69%** test accuracy on the 3333-signal dataset), a key weakness was identified in distinguishing **QPSK** under carrier frequency offsets ($\text{CFO} \ge 40\text{ kHz}$) and low SNR.

Under continuous carrier rotation $\Delta f$, the absolute phase trajectory rotates at a rate of $\frac{d\theta}{dt} = 2\pi \Delta f$, which smears the four constellation quadrants across time and mimics the frequency deviation trajectories of FSK.

While an initial 5-feature scalar differential-phase experiment (`run_007`) improved QPSK recall from $65.33\%$ to $76.00\%$, it introduced a severe performance regression on BPSK and MIXED ($80.77\%$ overall accuracy). Investigation showed that arithmetic window averaging of wrapped scalar angles $[-\pi, \pi]$ suffered from destructive branch-cut cancellations near $\pm\pi$.

To solve this, a **6-Feature Circular Differential Phasor** representation $[\cos(\Delta\theta), \sin(\Delta\theta)]$ has been implemented, providing mathematically exact circular averaging and phase-coherence encoding without branch-cut artifacts.

---

## 2. Feature Representation Comparison

| Metric | Baseline (4-Feature) | Failed Experiment (5-Feature Scalar) | New Enhanced (6-Feature Circular) |
|---|---|---|---|
| **Feature Channels** | $[I, Q, |s|, \theta]$ | $[I, Q, |s|, \theta, \Delta\theta]$ | **$[I, Q, |s|, \theta, \cos(\Delta\theta), \sin(\Delta\theta)]$** |
| **Token Representation** | $(256, 4)$ per signal | $(256, 5)$ per signal | **$(256, 6)$ per signal** |
| **Batch Tensor Shape** | $(\text{batch\_size}, 256, 4)$ | $(\text{batch\_size}, 256, 5)$ | **$(\text{batch\_size}, 256, 6)$** |
| **Window Aggregation** | Linear mean | Linear mean (broken for $\Delta\theta$) | **Circular mean resultant phasor (exact)** |
| **Number of Classes** | 5 (`BPSK`, `QPSK`, `FSK`, `QAM16`, `MIXED`) | 5 (`BPSK`, `QPSK`, `FSK`, `QAM16`, `MIXED`) | **5 (`BPSK`, `QPSK`, `FSK`, `QAM16`, `MIXED`)** |

---

## 3. Mathematical Formulation

For complex baseband signal $x[n] = I[n] + j Q[n]$:

The local conjugate product directly computes the differential rotation phasor:
$$P[n] = x[n] \cdot x^*[n-1]$$

Normalizing onto the unit circle for all valid samples ($|P[n]| > 10^{-12}$):
$$u[n] = \frac{P[n]}{|P[n]|}$$

The two circular differential features are:
$$\cos(\Delta\theta[n]) = \operatorname{Re}(u[n]), \quad \sin(\Delta\theta[n]) = \operatorname{Im}(u[n])$$

### Key Properties:
1. **No Modulo-$2\pi$ Branch Cuts**: Trigonometric components $(\cos, \sin)$ are smooth, bounded in $[-1, 1]$, and continuous everywhere on the circle.
2. **True Circular Mean Aggregation**: Window-averaging $\left(\frac{1}{K}\sum \cos(\Delta\theta), \frac{1}{K}\sum \sin(\Delta\theta)\right)$ computes the circular first trigonometric moment.
3. **Phase Coherence Encoding**: The length of the token's resultant phasor $R = \sqrt{\bar{\cos}^2 + \bar{\sin}^2} \in [0, 1]$ directly measures signal phase consistency across each token window.
4. **CFO Separation**: For a constant carrier offset $\Delta f$, the background rotation manifests as a constant direction $(\cos(2\pi\Delta f / f_s), \sin(2\pi\Delta f / f_s))$, allowing Transformer attention heads to isolate modulation phase transitions without destructive distortion.

---

## 4. Stability & Zero-Sample Handling

- **Boundary Condition ($n=0$)**: Initialized to unit phasor $\cos=1.0, \sin=0.0$.
- **Zero/Silence Handling**: Samples where $|P[n]| \le 10^{-12}$ map to $(0.0, 0.0)$ with `np.nan_to_num()` guaranteeing zero NaN/Inf values.
- **Dynamic Checkpoint Compatibility**: Checkpoints store metadata `input_features`. Evaluation and inference modules automatically detect and slice feature dimensions when loading legacy 4-feature checkpoints or new 6-feature checkpoints.

---

## 5. Files Changed

1. **[ai/features/learned_features.py](file:///c:/Users/HP/Desktop/SYNAPS/ai/features/learned_features.py)**:
   - Implemented 6-feature circular differential phasor $[\cos(\Delta\theta), \sin(\Delta\theta)]$ in `prepare_iq_features()`.
   - Updated `tokenize_signal_features()` docstrings for $(256, 6)$ token outputs.
2. **[ai/models/transformer.py](file:///c:/Users/HP/Desktop/SYNAPS/ai/models/transformer.py)**:
   - Updated `SignalTransformer` default `input_features=6`.
3. **[ai/training/train.py](file:///c:/Users/HP/Desktop/SYNAPS/ai/training/train.py)**:
   - Set `INPUT_FEATURES = 6`.
   - Updated model instantiation and dynamic checkpoint saving metadata.
4. **[ai/training/evaluate.py](file:///c:/Users/HP/Desktop/SYNAPS/ai/training/evaluate.py)**:
   - Updated checkpoint loader to detect `in_features` dynamically (backward compatible with 4-feature and 6-feature models).
5. **[ai/inference/predict.py](file:///c:/Users/HP/Desktop/SYNAPS/ai/inference/predict.py)**:
   - Updated checkpoint loader and inference feature preparation with dynamic dimension slicing.
6. **[tests/ai/test_ai.py](file:///c:/Users/HP/Desktop/SYNAPS/tests/ai/test_ai.py)**:
   - Updated unit tests and assertions for 6-feature tensors.

---

## 6. Verification & Test Results

- **Unit Test Suite**: All tests in `tests/ai/test_ai.py` pass cleanly.
- **Feature Robustness Test**: Handles raw signals, short signals, zero signals, and high-CFO signals safely with finite bounds.
- **Smoke Test Forward/Backward**: Passes forward propagation, loss calculation, and backward gradient computation.
- **Baseline Integrity**: The 4-feature baseline checkpoint (`ai/models/transformer.pth`) and baseline run artifacts (`result/transformer/run_006.*`) are preserved.
