"""
SYNAPS Copilot — Knowledge Base
Static RF / signal theory and SYNAPS platform knowledge.
"""

SYNAPS_KNOWLEDGE = """
SYNAPS is an autonomous RF signal intelligence platform that performs:
- Automatic Modulation Recognition (AMR) using a Transformer-based deep learning model
- Digital Signal Processing (DSP): IQ signal parsing, FFT, PSD, spectrogram generation
- Signal feature extraction: SNR estimation, bandwidth, carrier frequency offset (CFO),
  symbol rate, higher-order cumulants (C40, C42), PAPR, RF fingerprinting
- Bit recovery and ground-truth validation (BER computation, decoded payload)
- Visualization: time-domain waveform, frequency spectrum (PSD), time-frequency spectrogram
- Multi-carrier / MIXED signal detection and component separation

SYNAPS supports modulation classes: BPSK, QPSK, FSK, QAM16

WORKFLOW:
1. User uploads an IQ (.iq) or WAV (.wav) file on the Analyze page, or selects a dataset sample
2. The frontend sends the file to the FastAPI backend via POST /analysis/upload-and-analyze
3. The backend runs the full DSP + AI pipeline via SYNAPS Signal Pipeline
4. Results are displayed on the Results page: classification, metrics, waveform, spectrum, spectrogram, features, explanation
5. Authenticated users have their analysis results saved automatically to the Supabase database

PAGES:
- "/" (Home/Landing): Introduction to SYNAPS, key features, how it works
- "/analyze" (Analyze): Upload IQ/WAV file or select dataset sample for analysis — requires login
- "/processing" (Processing): Shows analysis progress while backend runs
- "/results" (Results): Full detailed analysis report with charts, metrics, and DSP features
- "/dashboard" (Dashboard): User analytics — history charts, top modulations, recent analyses
- "/history" (History): Full paginated list of past analysis reports with search and filter
- "/admin" (Admin): Administrator dashboard — platform-wide stats, user management (admin role only)
- "/auth" (Auth): Sign in / sign up page

AUTHENTICATION:
- SYNAPS uses Supabase email/password authentication
- Sign in at /auth
- User must be authenticated to run analyses
- Analysis results are stored per user in the analysis_reports table
- Admins have role="admin" in the profiles table

DATABASE TABLE: analysis_reports
Fields: id, user_id, created_at, filename, format, classification, confidence, sample_rate,
        duration, bandwidth, snr, peak_frequency, num_samples, prediction_breakdown,
        features, explanation, raw_report
"""

RF_SIGNAL_THEORY = """
RF AND SIGNAL THEORY KNOWLEDGE:

IQ SIGNALS:
- I/Q (In-phase / Quadrature) signals are the standard representation for RF signals
- A complex IQ sample encodes both amplitude and phase: s(t) = I(t) + jQ(t)
- IQ files contain raw complex-valued samples captured from radio hardware (SDR, spectrum analyzers, etc.)
- Common IQ formats: binary 32-bit float pairs (I, Q interleaved)

SAMPLING & NYQUIST:
- Sample rate (Fs): number of IQ samples captured per second; typically in MHz
- Nyquist theorem: to faithfully represent a signal of bandwidth B, the sample rate must be at least 2B
- Aliasing occurs if the sample rate is too low, folding high-frequency content into the baseband
- Duration = number of samples / sample rate

FREQUENCY DOMAIN:
- FFT (Fast Fourier Transform): converts time-domain samples to frequency-domain representation
- Frequency spectrum: shows power vs. frequency; reveals carrier, harmonics, sidebands
- PSD (Power Spectral Density): power per unit frequency (dBm/Hz or normalized)
- Occupied Bandwidth (99% Power): bandwidth containing 99% of signal power; measures spectral efficiency
- Carrier Frequency Offset (CFO): difference between nominal and actual carrier frequency; caused by oscillator drift

SPECTROGRAM:
- Time-frequency representation computed by applying FFT to overlapping time windows (STFT)
- X-axis: time; Y-axis: frequency; color: power
- Reveals how signal spectrum evolves over time; useful for burst detection and frequency hopping

SNR:
- Signal-to-Noise Ratio: ratio of signal power to noise power, typically in dB
- Higher SNR → cleaner, more reliable signal → higher classification confidence
- Low SNR can degrade modulation recognition accuracy
- Estimated via noise floor estimation in the frequency domain

MODULATION TYPES:
- BPSK (Binary Phase Shift Keying): 2 constellation points (0°, 180°); 1 bit/symbol; simplest, most robust
- QPSK (Quadrature Phase Shift Keying): 4 constellation points (45°, 135°, 225°, 315°); 2 bits/symbol
- FSK (Frequency Shift Keying): encodes bits as different carrier frequencies; constant amplitude
- QAM16 (16-Quadrature Amplitude Modulation): 16 constellation points; 4 bits/symbol; requires higher SNR
- Higher-order modulations (QAM64, 256-QAM) pack more bits but require excellent SNR

HIGHER-ORDER CUMULANTS:
- Statistical measures used in blind modulation classification
- C40 (4th-order cumulant): discriminates between modulations; BPSK ≈ 2.0, QPSK ≈ -1.0, QAM16 ≈ -0.68
- C42 (4th-order, mixed cumulant): further discriminates modulation type and order
- Cumulants are robust to Gaussian noise (cumulants of Gaussian are zero above 2nd order)

PAPR (Peak-to-Average Power Ratio):
- Ratio of peak signal power to average power
- High PAPR (e.g., QAM16) requires linear amplifiers to avoid distortion
- BPSK and FSK have low PAPR; multi-carrier signals (OFDM) have very high PAPR

SYMBOL RATE:
- Number of symbols (modulation states) transmitted per second (Baud)
- Bit rate = Symbol rate × bits per symbol

SYNCHRONIZATION:
- Carrier synchronization: recovering the exact carrier frequency and phase
- Timing synchronization: recovering the exact sampling instants for symbol boundaries
- SYNAPS performs blind synchronization for bit recovery

BIT RECOVERY:
- After classification and synchronization, SYNAPS demodulates the signal to recover raw bits
- Reference bits (from ground truth metadata) are compared to recovered bits
- Bit accuracy = matched_bits / total_bits × 100%
- BER (Bit Error Rate) = bit_errors / total_bits; should be < 0.001 for good links
- Validation status: VALIDATED = accurate recovery, otherwise provides status description

RF FINGERPRINTING:
- Each radio transmitter has subtle hardware imperfections (IQ imbalance, oscillator noise, PA nonlinearity)
- These create a unique "fingerprint" in the signal even for identical hardware models
- SYNAPS computes an RF fingerprint ID hash from extracted hardware imperfection features
- Used for emitter identification beyond protocol-level classification

TRANSFORMER-BASED MODULATION CLASSIFICATION:
- SYNAPS uses a Transformer neural network (attention-based) trained on IQ sequences
- Input: raw IQ samples or feature vectors extracted from DSP processing
- The model outputs probability distributions across classes (BPSK, QPSK, FSK, QAM16)
- Confidence = max probability in the softmax output
- High confidence (>90%) indicates unambiguous modulation detection
- Moderate confidence (60–90%) suggests noise or multi-carrier interference

CONSTELLATION:
- Geometric plot of I vs Q values showing symbol locations
- Each modulation type has a characteristic constellation pattern
- BPSK: 2 points on I-axis; QPSK: 4 diagonal points; QAM16: 4×4 grid
"""

NAVIGATION_GUIDE = """
HOW TO USE SYNAPS:

TO ANALYZE A SIGNAL:
1. Sign in at /auth (or click "Sign in" in the header)
2. Click "Analyze" in the navigation or go to /analyze
3. Upload a .iq or .wav file using the Upload tab, or choose a dataset sample from the Dataset tab
4. Click "Run Analysis" to start processing
5. Wait for processing to complete (shown on the /processing page)
6. View the full results on /results

TO VIEW HISTORY:
- Click "History" in the navigation (requires login)
- Shows all your past analyses with filtering by modulation type and search by filename
- Click any entry to see details

TO VIEW DASHBOARD:
- Click "Dashboard" in the navigation (requires login)
- Shows summary charts: analysis activity over time, modulation distribution, confidence bars, signal quality

TO USE ADMIN:
- Only accessible to users with admin role
- Shows platform-wide statistics and user management

SUPPORTED FILE FORMATS:
- .iq (binary IQ file: interleaved float32 I/Q samples)
- .wav (WAV file with IQ-like audio; sample rate is auto-detected)
"""

PAGE_CONTEXTS = {
    "home": "User is on the SYNAPS home/landing page. Focus on explaining what SYNAPS is, how it works, and what it can do.",
    "analyze": "User is on the Analyze page. They are preparing to run a signal analysis — uploading a file or selecting a dataset sample.",
    "processing": "User is on the Processing page. Their signal is currently being analyzed by the SYNAPS backend.",
    "results": "User is viewing a completed signal analysis result. Help them understand the classification, metrics, visualizations, and DSP features in context of their specific result.",
    "dashboard": "User is on their Dashboard. Focus on helping them understand their analysis history and statistics.",
    "history": "User is viewing their analysis history list. Help with navigation and understanding past results.",
    "admin": "User is on the Admin Dashboard. Provide general navigation help only. Do not expose user data or system internals.",
    "unknown": "User is on an unspecified page of SYNAPS.",
}

QUICK_PROMPTS = {
    "home": [
        "What is SYNAPS?",
        "How does signal analysis work?",
        "What is an IQ signal?",
        "What modulations does SYNAPS support?",
    ],
    "analyze": [
        "How do I analyze a signal?",
        "What file formats are supported?",
        "What does SNR mean?",
        "Explain the analysis pipeline",
    ],
    "results": [
        "Explain this result",
        "Why was this modulation detected?",
        "What does confidence mean?",
        "What is the spectrogram showing?",
    ],
    "dashboard": [
        "Summarize my analyses",
        "What modulation do I see most?",
        "What was my latest analysis?",
        "How many signals have I analyzed?",
    ],
    "history": [
        "How do I search my history?",
        "What was my last QAM16 analysis?",
        "Explain how to filter by modulation",
    ],
    "processing": [
        "How long does analysis take?",
        "What is SYNAPS computing right now?",
        "What happens after processing?",
    ],
    "unknown": [
        "What is SYNAPS?",
        "How do I analyze a signal?",
        "What is QAM16?",
    ],
}
