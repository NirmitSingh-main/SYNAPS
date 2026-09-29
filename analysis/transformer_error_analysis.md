# Transformer Error Analysis

## 1. Current Model

| Property | Value |
|---|---|
| **Checkpoint Path** | [ai/models/transformer.pth](file:///c:/Users/HP/Desktop/SYNAPS/ai/models/transformer.pth) |
| **Best Checkpoint Epoch** | Epoch 44 (of 50 trained) |
| **Best Validation Accuracy** | 87.05% |
| **Test Accuracy** | **87.69%** (342 / 390 correct) |
| **Test Loss** | 0.3304 |
| **Input Representation** | Fixed 256-token window-averaged feature matrix `(batch_size, 256, 4)` ($I$, $Q$, magnitude $|s|$, phase $\angle s$) |
| **Raw Signal Lengths** | 8,000 samples (pure) / 9,600 samples (mixed) |
| **Number of Classes** | 5 (`BPSK`, `QPSK`, `FSK`, `QAM16`, `MIXED`) |
| **Evaluation Device** | CUDA (NVIDIA GeForce RTX 4050 Laptop GPU) |
| **Evaluation Records** | [analysis/transformer_misclassifications.csv](file:///c:/Users/HP/Desktop/SYNAPS/analysis/transformer_misclassifications.csv) |

---

## 2. Confusion Analysis

### 5×5 Confusion Matrix (Rows = Actual, Columns = Predicted)

| Actual \ Predicted | BPSK | QPSK | FSK | QAM16 | MIXED | Total Actual | Total Errors (FN) | Class Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BPSK** | **65** | 8 | 1 | 1 | 0 | 75 | 10 | 86.67% |
| **QPSK** | 10 | **49** | 13 | 1 | 2 | 75 | 26 | 65.33% |
| **FSK** | 2 | 4 | **69** | 0 | 0 | 75 | 6 | 92.00% |
| **QAM16** | 3 | 0 | 0 | **72** | 0 | 75 | 3 | 96.00% |
| **MIXED** | 0 | 0 | 3 | 0 | **87** | 90 | 3 | 96.67% |
| **Total Predicted** | 80 | 61 | 86 | 74 | 89 | 390 | 48 | — |
| **False Positives (FP)** | 15 | 12 | 17 | 2 | 2 | — | 48 | — |
| **Class Precision** | 81.25% | 80.33% | 80.23% | 97.30% | 97.75% | — | — | **87.69% Avg** |

### Per-Class Performance Summary

| Class | True Count | Predicted Count | True Positives | False Negatives | False Positives | Recall | Precision | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BPSK** | 75 | 80 | 65 | 10 | 15 | 86.67% | 81.25% | 83.87% |
| **QPSK** | 75 | 61 | 49 | 26 | 12 | **65.33%** | 80.33% | 72.06% |
| **FSK** | 75 | 86 | 69 | 6 | 17 | 92.00% | 80.23% | 85.71% |
| **QAM16** | 75 | 74 | 72 | 3 | 2 | 96.00% | 97.30% | 96.64% |
| **MIXED** | 90 | 89 | 87 | 3 | 2 | 96.67% | 97.75% | 97.21% |
| **Overall** | **390** | **390** | **342** | **48** | **48** | **87.69%** | **87.69%** | **87.10%** |

### Most Common Confusion Pairs

| Rank | True Class | Predicted Class | Error Count | Share of Total Errors (48) | Primary Root Factors Observed |
|:---:|:---:|:---:|:---:|:---:|---|
| 1 | **QPSK** | **FSK** | **13** | 27.08% | Severe CFO + low/medium SNR causing apparent frequency shift |
| 2 | **QPSK** | **BPSK** | **10** | 20.83% | Low SNR (<10 dB) & phase smearing collapsing 4-QAM constellation into 1D |
| 3 | **BPSK** | **QPSK** | **8** | 16.67% | High CFO (>40 kHz) generating apparent quadrature rotation |
| 4 | **FSK** | **QPSK** | **4** | 8.33% | Low SNR (mean 5.79 dB) + timing offsets |
| 5 | **QAM16** | **BPSK** | **3** | 6.25% | Extremely low SNR (mean 2.5 dB) obscuring multi-level amplitude grid |
| 5 | **MIXED** | **FSK** | **3** | 6.25% | Multi-carrier frequency components resembling FSK tone switching |
| 7 | **FSK** | **BPSK** | **2** | 4.17% | Low SNR (<3 dB) + high CFO |
| 7 | **QPSK** | **MIXED** | **2** | 4.17% | Very low SNR (0.08 dB, 2.65 dB) + extreme CFO (~-48 kHz) mimicking composite spectra |
| 9 | **BPSK** | **FSK** | **1** | 2.08% | Very low SNR (1.1 dB) + high CFO (45.9 kHz) |
| 9 | **BPSK** | **QAM16** | **1** | 2.08% | High CFO (45.6 kHz) + roll-off variations |
| 9 | **QPSK** | **QAM16** | **1** | 2.08% | Moderate SNR with low SPS |

> **Key Observation**: QPSK accounts for **26 out of 48 total errors (54.17%)**. Its recall (65.33%) is significantly lower than all other classes ($86.67\% - 96.67\%$).

---

## 3. QPSK Error Analysis

QPSK misclassifications fall into four distinct confusion directions:

### A. QPSK → FSK (13 samples, 27.1% of all errors)
- **Characteristics**:
  - Average SNR: $13.28\text{ dB}$ (ranging from $3.47\text{ dB}$ to $28.28\text{ dB}$).
  - Average $|\text{CFO}|$: $25,832\text{ Hz}$ (up to $49,370\text{ Hz}$).
  - Symbol Rate: Mean $138.5\text{ kHz}$ (higher than the general QPSK average of $127.5\text{ kHz}$).
  - Roll-off factor: Mean $0.360$ (higher than correct QPSK mean $0.294$).
- **Mechanism**:
  - When QPSK experiences uncompensated frequency offset $\Delta f$, the phase rotates at rate $\frac{d\theta}{dt} = 2\pi \Delta f$.
  - In the 256-token window aggregation, continuous linear phase ramp across time windows mimics the alternating positive/negative instantaneous frequency trajectories characteristic of FSK tone shifts.

### B. QPSK → BPSK (10 samples, 20.8% of all errors)
- **Characteristics**:
  - Average SNR: **$12.54\text{ dB}$** (median **$8.87\text{ dB}$**, min **$0.64\text{ dB}$**). Significantly lower than correct QPSK ($17.55\text{ dB}$).
  - Average $|\text{CFO}|$: $28,142\text{ Hz}$.
  - Samples Per Symbol (SPS): Mean $11.9$ (vs $10.2$ for correct QPSK).
- **Mechanism**:
  - At low SNR, Gaussian noise clouds merge adjacent constellation points.
  - In feature space $[I, Q, |s|, \angle s]$, the token-averaged variance across the quadrature axis gets masked by noise, causing the network to recognize only the dominant binary polarity (BPSK).

### C. QPSK → MIXED (2 samples)
- **Characteristics**:
  - `signal_00792_qpsk`: $\text{SNR} = 2.65\text{ dB}$, $\text{CFO} = -47,467\text{ Hz}$, $\text{Confidence} = 0.3017$ (lowest confidence in dataset).
  - `signal_00830_qpsk`: $\text{SNR} = 0.08\text{ dB}$, $\text{CFO} = -48,886\text{ Hz}$, $\text{Confidence} = 0.5975$.
- **Mechanism**:
  - Near $0\text{ dB}$ SNR combined with near-Nyquist CFO ($|\text{CFO}| \approx 48\text{ kHz}$) causes significant spectral distortion and noise floor elevation, which the Transformer interprets as a multi-signal overlapping spectrum (MIXED).

### D. QPSK → QAM16 (1 sample)
- **Characteristics**:
  - `signal_00818_qpsk`: $\text{SNR} = 22.37\text{ dB}$, $\text{CFO} = -45,574\text{ Hz}$, $\text{SPS} = 20$, $\text{Roll-off} = 0.193$, $\text{Confidence} = 0.8122$.
- **Mechanism**:
  - High SPS ($20$) with sharp roll-off ($0.193$) creates significant inter-symbol transition amplitude ripple, which mimics the multi-amplitude levels of 16-QAM.

---

## 4. BPSK/FSK Confusion Analysis

### BPSK Misclassifications (10 samples)
1. **BPSK → QPSK (8 samples)**:
   - Mean SNR: $16.92\text{ dB}$ (spread across all SNR levels).
   - Mean $|\text{CFO}|$: $29,214\text{ Hz}$ (with 4 samples having $|\text{CFO}| > 45\text{ kHz}$).
   - Roll-off: Mean $0.365$ (vs $0.284$ for correct BPSK).
   - **Mechanism**: High CFO causes continuous rotation of the 1D BPSK constellation into the Quadrature ($Q$) plane, populating both $I$ and $Q$ channels and mimicking a 4-phase constellation.
2. **BPSK → FSK (1 sample)**:
   - `signal_00519_bpsk`: $\text{SNR} = 1.11\text{ dB}$, $\text{CFO} = 45,954\text{ Hz}$, $\text{Conf} = 0.4710$.
3. **BPSK → QAM16 (1 sample)**:
   - `signal_00142_bpsk`: $\text{SNR} = 13.56\text{ dB}$, $\text{CFO} = 45,563\text{ Hz}$, $\text{Conf} = 0.9586$.

### FSK Misclassifications (6 samples)
1. **FSK → QPSK (4 samples)**:
   - Mean SNR: **$5.79\text{ dB}$** (median $5.39\text{ dB}$, range $1.86 - 10.53\text{ dB}$).
   - Mean $|\text{CFO}|$: $27,088\text{ Hz}$.
   - Filter span: Exactly $4.0$ symbols (vs mean $5.57$ for correct FSK).
   - **Mechanism**: Under severe noise ($\text{SNR} < 6\text{ dB}$), instantaneous frequency deviations are obscured by noise phase jitter, resembling random phase transitions around the origin.
2. **FSK → BPSK (2 samples)**:
   - `signal_01509_fsk`: $\text{SNR} = 1.93\text{ dB}$, $\text{CFO} = 34,705\text{ Hz}$, $\text{Conf} = 0.6393$.
   - `signal_01996_fsk`: $\text{SNR} = 0.58\text{ dB}$, $\text{CFO} = 48,169\text{ Hz}$, $\text{Conf} = 0.4320$.

---

## 5. MIXED Error Analysis

The MIXED class demonstrates strong performance: **$87 / 90$ correct ($96.67\%$ accuracy)**.

### Detailed Breakdown of MIXED Test Samples

| Component Category | Total Test Samples | Correct | Misclassified (as FSK) | Accuracy |
|---|:---:|:---:|:---:|:---:|
| **2 Components Total** | **59** | **57** | **2** | **96.61%** |
| • QPSK + QAM16 | 38 | 36 | 2 | 94.74% |
| • FSK + QAM16 | 21 | 21 | 0 | 100.00% |
| **3 Components Total** | **31** | **30** | **1** | **96.77%** |
| • BPSK + QPSK + FSK | 16 | 15 | 1 | 93.75% |
| • QPSK + FSK + QAM16 | 15 | 15 | 0 | 100.00% |
| **Total MIXED** | **90** | **87** | **3** | **96.67%** |

### Detailed Inspection of the 3 MIXED Misclassifications

1. **`signal_03248_mixed`**:
   - Components: 2 (`QPSK`, `QAM16`)
   - $\text{SNR} = 16.47\text{ dB}$
   - Predicted: `FSK` ($\text{Confidence} = 0.7660$)
   - Output Probabilities: $\text{FSK} = 76.60\%$, $\text{MIXED} = 22.84\%$, others $< 0.5\%$
2. **`signal_03265_mixed`**:
   - Components: 2 (`QPSK`, `QAM16`)
   - $\text{SNR} = 26.39\text{ dB}$
   - Predicted: `FSK` ($\text{Confidence} = 0.9642$)
   - Output Probabilities: $\text{FSK} = 96.42\%$, $\text{MIXED} = 3.52\%$
3. **`signal_03309_mixed`**:
   - Components: 3 (`BPSK`, `QPSK`, `FSK`)
   - $\text{SNR} = 9.63\text{ dB}$
   - Predicted: `FSK` ($\text{Confidence} = 0.5596$)
   - Output Probabilities: $\text{FSK} = 55.96\%$, $\text{MIXED} = 43.83\%$

### Comparison of 2-Component vs 3-Component Signals
- **2-component signals**: $96.61\%$ accuracy ($2$ errors out of $59$).
- **3-component signals**: $96.77\%$ accuracy ($1$ error out of $31$).
- **Finding**: There is **no statistically significant difference** in classification accuracy between 2-component and 3-component mixed signals ($p > 0.9$). All 3 misclassifications were assigned to `FSK`.

---

## 6. Confidence Analysis

### Summary Statistics

| Metric | Correct Predictions ($n=342$) | Misclassifications ($n=48$) |
|---|:---:|:---:|
| **Mean Confidence** | **93.99%** | **71.07%** |
| **Median Confidence** | **99.59%** | **68.89%** |
| **Standard Deviation** | 11.61% | 18.35% |
| **Minimum Confidence** | 39.29% | 30.17% |
| **Maximum Confidence** | 99.99% | 98.27% |
| **25th Percentile** | 94.67% | 58.21% |
| **75th Percentile** | 99.93% | 85.93% |

### Confidence Distribution of Errors

| Confidence Range | Count | Share of Errors | Interpretation for Decision Engine |
|---|:---:|:---:|---|
| **$\ge 90\%$** | 11 | 22.92% | High-confidence false alarms (over-confident under high CFO) |
| **$80\% - 90\%$** | 6 | 12.50% | Moderate-high confidence errors |
| **$70\% - 80\%$** | 6 | 12.50% | Near baseline threshold |
| **$50\% - 70\%$** | 19 | 39.58% | **Low-confidence uncertain zone (can be flagged as UNKNOWN/ambiguous)** |
| **$< 50\%$** | 6 | 12.50% | **Very low confidence (correctly rejected by $\ge 50\%$ threshold)** |

### Top 5 Highest-Confidence Errors
1. `signal_00756_qpsk` (QPSK → BPSK): $\text{Conf} = 98.27\%$, $\text{SNR} = 3.7\text{ dB}$, $\text{CFO} = -24,869\text{ Hz}$
2. `signal_00785_qpsk` (QPSK → BPSK): $\text{Conf} = 98.21\%$, $\text{SNR} = 5.9\text{ dB}$, $\text{CFO} = 24,208\text{ Hz}$
3. `signal_00768_qpsk` (QPSK → BPSK): $\text{Conf} = 97.81\%$, $\text{SNR} = 26.3\text{ dB}$, $\text{CFO} = 37,197\text{ Hz}$
4. `signal_01347_qpsk` (QPSK → FSK): $\text{Conf} = 97.72\%$, $\text{SNR} = 17.8\text{ dB}$, $\text{CFO} = -49,370\text{ Hz}$
5. `signal_03265_mixed` (MIXED → FSK): $\text{Conf} = 96.42\%$, $\text{SNR} = 26.4\text{ dB}$

### Top 5 Lowest-Confidence Errors
1. `signal_00792_qpsk` (QPSK → MIXED): $\text{Conf} = 30.17\%$, $\text{SNR} = 2.65\text{ dB}$, $\text{CFO} = -47,467\text{ Hz}$
2. `signal_00589_bpsk` (BPSK → QPSK): $\text{Conf} = 33.14\%$, $\text{SNR} = 1.71\text{ dB}$, $\text{CFO} = 47,427\text{ Hz}$
3. `signal_01194_qpsk` (QPSK → BPSK): $\text{Conf} = 39.69\%$, $\text{SNR} = 2.11\text{ dB}$, $\text{CFO} = -30,592\text{ Hz}$
4. `signal_01996_fsk` (FSK → BPSK): $\text{Conf} = 43.20\%$, $\text{SNR} = 0.58\text{ dB}$, $\text{CFO} = 48,169\text{ Hz}$
5. `signal_01630_fsk` (FSK → QPSK): $\text{Conf} = 43.49\%$, $\text{SNR} = 1.86\text{ dB}$, $\text{CFO} = -25,599\text{ Hz}$

> **Intelligence Fusion Value**: Over **$52.1\%$ of errors** fall below the standard $0.70$ confidence threshold, allowing the hybrid DSP-AI decision arbiter in SYNAPS to safely defer or cross-validate with cyclostationary/higher-order cumulant features.

---

## 7. Signal-Parameter Analysis

### A. Signal-to-Noise Ratio (SNR) Impact
Evaluation across pure signal test samples ($n=300$):

| SNR Range | Total Samples | Correct | Misclassified | Accuracy |
|---|:---:|:---:|:---:|:---:|
| **$\text{SNR} < 5\text{ dB}$** | **37** | **22** | **15** | **59.46%** |
| **$5\text{ dB} \le \text{SNR} < 10\text{ dB}$** | 56 | 48 | 8 | **85.71%** |
| **$10\text{ dB} \le \text{SNR} < 20\text{ dB}$** | 100 | 87 | 13 | **87.00%** |
| **$20\text{ dB} \le \text{SNR} < 30\text{ dB}$** | 97 | 88 | 9 | **90.72%** |
| **$\text{SNR} \ge 30\text{ dB}$** | 10 | 10 | 0 | **100.00%** |

- **Finding**: Error rate is strongly inversely correlated with SNR. Below $5\text{ dB}$, error rate jumps to $40.5\%$. Above $30\text{ dB}$, accuracy is $100\%$.

### B. Carrier Frequency Offset ($|\text{CFO}|$) Impact
Evaluation across pure signal test samples ($n=300$):

| $|\text{CFO}|$ Range | Total Samples | Correct | Misclassified | Accuracy |
|---|:---:|:---:|:---:|:---:|
| **$0 - 10\text{ kHz}$** | 65 | 55 | 10 | **84.62%** |
| **$10 - 25\text{ kHz}$** | 89 | 80 | 9 | **89.89%** |
| **$25 - 40\text{ kHz}$** | 89 | 79 | 10 | **88.76%** |
| **$\ge 40\text{ kHz}$** | **57** | **41** | **16** | **71.93%** |

- **Finding**: Accuracy remains steady ($\sim 85-90\%$) for $|\text{CFO}| \le 40\text{ kHz}$, but drops sharply to **$71.93\%$** when $|\text{CFO}| \ge 40\text{ kHz}$. Extreme CFO induces fast phase trajectory rotation that fools phase-based token representations.

### C. Samples Per Symbol (SPS) & Symbol Rate
| Samples Per Symbol | Total Samples | Correct | Misclassified | Accuracy |
|---|:---:|:---:|:---:|:---:|
| **SPS = 5** ($f_{sym}=200\text{ kHz}$) | 36 | 29 | 7 | 80.56% |
| **SPS = 8** ($f_{sym}=125-150\text{ kHz}$) | 110 | 95 | 15 | 86.36% |
| **SPS = 10** ($f_{sym}=100\text{ kHz}$) | 43 | 38 | 5 | 88.37% |
| **SPS = 16** ($f_{sym}=75\text{ kHz}$) | 77 | 68 | 9 | 88.31% |
| **SPS = 20** ($f_{sym}=50\text{ kHz}$) | 34 | 25 | 9 | 73.53% |

### D. Other Signal Parameters
- **Phase Offset ($\theta_0$)**: Evenly distributed ($0^\circ - 360^\circ$) across both correct and error sets (correct mean $187.4^\circ$, error mean $172.9^\circ$). No significant bias.
- **Amplitude Scale**: Correct mean $1.02 \pm 0.29$, error mean $1.04 \pm 0.28$. Amplitude scaling does not correlate with error rates because input features are magnitude-normalized.
- **Sampling Frequency ($1.0\text{ MHz}$ vs $1.2\text{ MHz}$)**:
  - $1.0\text{ MHz}$: $177$ correct / $203$ total ($87.19\%$)
  - $1.2\text{ MHz}$: $165$ correct / $187$ total ($88.24\%$)
  - No significant difference between sampling rates.
- **Frequency Placement**: All pure signals in the dataset have nominal `frequency_placement_hz = 0.0`.

---

## 8. Evidence-Based Findings

1. **QPSK is the Primary Failure Mode**:
   - QPSK accounts for $54.2\%$ of all classification errors ($26/48$), achieving only $65.33\%$ recall.
   - Its most frequent confusions are `QPSK -> FSK` ($13$ cases) and `QPSK -> BPSK` ($10$ cases).

2. **Dual-Factor Error Driver: Low SNR + High CFO**:
   - Severe noise ($\text{SNR} < 5\text{ dB}$) reduces classification accuracy to $59.46\%$.
   - High frequency offset ($|\text{CFO}| \ge 40\text{ kHz}$) reduces accuracy to $71.93\%$.
   - When low SNR and high CFO coincide, QPSK constellations undergo both radial noise diffusion and rotational smearing, making them statistically closer to FSK or BPSK.

3. **MIXED Classification is Highly Robust**:
   - MIXED signals achieved **$96.67\%$ accuracy** ($87/90$).
   - The tokenization process preserves multi-component spectral signatures effectively.
   - Both 2-component ($96.61\%$) and 3-component ($96.77\%$) signals classify with equal reliability.

4. **Confidence Thresholding is Viable for Fusion**:
   - Correct samples exhibit an average confidence of $93.99\%$ (median $99.59\%$).
   - Over $52\%$ of errors exhibit confidence $< 0.70$, confirming that the model's Softmax entropy serves as a meaningful quality indicator for downstream DSP-AI fusion.

5. **Parameters with Insufficient Evidence**:
   - *Phase offset*, *amplitude scaling*, and *sampling frequency* show no statistically significant correlation with error occurrences.
   - *Insufficient evidence from the current test set* to attribute errors to pulse shaping filter span or initial phase.

---

## 9. Possible Future Experiments

*(Listed for future consideration only — not implemented)*

1. **CFO-Robust Feature Augmentation**:
   - Experiment with adding differential phase $\Delta \theta[n] = \theta[n] - \theta[n-1]$ or instantaneous frequency features to make representations invariant to static carrier frequency offsets.

2. **Multiscale Tokenization / Dual Resolution Aggregation**:
   - Experiment with combining 128 coarse window tokens with 512 fine sub-tokens to preserve both high-frequency phase transitions and long-term envelope statistics.

3. **Pre-Transformer Coarse CFO Compensation**:
   - Experiment with running an FFT-based $M$-th power carrier frequency offset coarse estimator prior to feature normalization.

4. **Curriculum Training / Hard-Negative Mining on Low-SNR QPSK**:
   - Experiment with focal loss ($\gamma = 2.0$) or class-weighted cross-entropy to increase penalty on QPSK misclassifications.

5. **Higher-Order Cumulant Feature Injection**:
   - Experiment with concatenating normalized cumulants ($C_{40}, C_{42}$) to the Transformer embedding to explicitly separate QPSK ($C_{42} \neq 0, C_{40} \neq 0$) from BPSK and FSK.
