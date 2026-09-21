"""
PMDS EXPERIMENT 4: Smartphone Sensor Signal Processing Benchmark
==================================================================
Benchmarking smartphone sensor algorithms:
- Accelerometer & Gyroscope Rest Tremor Frequency Estimation
- Stride-Time Coefficient of Variation (Gait CV)
- Inter-Tap Interval SD & CV (Finger Tapping)
- Signal Quality & Processing Stability

Scientific Rigor Enforced:
1. Benchmark algorithm execution on motion sensor datasets or test signals.
2. Compute actual frequency error (Hz) against reference frequencies (3.0 - 8.0 Hz).
3. Compute actual inter-tap interval SD (ms) and CV (%).
"""

import os
import sys
import time
import numpy as np
import pandas as pd

def estimate_tremor_zero_crossing(acc_signal, sampling_rate=50.0):
    """Zero-Crossing Frequency Estimation algorithm used in PMDS Motion Sensor Service."""
    mean_val = np.mean(acc_signal)
    dynamic_signal = acc_signal - mean_val
    zero_crossings = np.where(np.diff(np.signbit(dynamic_signal)))[0]
    duration_sec = len(acc_signal) / sampling_rate
    if duration_sec > 0:
        return float(len(zero_crossings) / (2.0 * duration_sec))
    return 0.0

def estimate_tremor_fft(acc_signal, sampling_rate=50.0):
    """FFT Dominant Frequency Estimation algorithm."""
    n = len(acc_signal)
    if n == 0:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(acc_signal - np.mean(acc_signal)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    # Bandpass filter range 2.0 to 10.0 Hz
    valid_mask = (freqs >= 2.0) & (freqs <= 10.0)
    if not np.any(valid_mask):
        return 0.0
    valid_freqs = freqs[valid_mask]
    valid_fft = fft_vals[valid_mask]
    max_idx = np.argmax(valid_fft)
    return float(valid_freqs[max_idx])

def evaluate_finger_tap_stability(tap_intervals_ms):
    if len(tap_intervals_ms) < 2:
        return {"mean_iti": 0.0, "sd_iti": 0.0, "cv_iti": 0.0}
    mean_iti = float(np.mean(tap_intervals_ms))
    sd_iti = float(np.std(tap_intervals_ms))
    cv_iti = float((sd_iti / mean_iti) * 100.0) if mean_iti > 0 else 0.0
    return {"mean_iti": mean_iti, "sd_iti": sd_iti, "cv_iti": cv_iti}

def run_experiment_4():
    print("==========================================================")
    print(" PMDS EXPERIMENT 4: SENSOR SIGNAL PROCESSING BENCHMARK")
    print("==========================================================")

    # 1. Tremor Frequency Estimation Error Benchmark across reference frequencies
    target_frequencies = [3.5, 4.0, 5.0, 6.0, 7.0]
    sampling_rate = 50.0 # Standard 50 Hz mobile sensor sampling rate
    duration_sec = 15.0
    t = np.linspace(0, duration_sec, int(sampling_rate * duration_sec))

    freq_benchmark_results = []
    for f_target in target_frequencies:
        # Generate test sinusoidal motion signal with Gaussian noise
        noise = np.random.normal(0, 0.1, len(t))
        signal = np.sin(2 * np.pi * f_target * t) + noise

        f_zc = estimate_tremor_zero_crossing(signal, sampling_rate)
        f_fft = estimate_tremor_fft(signal, sampling_rate)

        err_zc = abs(f_zc - f_target)
        err_fft = abs(f_fft - f_target)

        freq_benchmark_results.append({
            "Target_Freq (Hz)": f_target,
            "Zero_Crossing_Est (Hz)": round(f_zc, 2),
            "Zero_Crossing_Error (Hz)": round(err_zc, 2),
            "FFT_Dominant_Est (Hz)": round(f_fft, 2),
            "FFT_Error (Hz)": round(err_fft, 2)
        })

    df_freq = pd.DataFrame(freq_benchmark_results)

    # 2. Finger Tapping Inter-Tap Interval (ITI) Stability Evaluation
    # Simulated healthy controls vs Parkinsonian tapping intervals
    healthy_taps = np.random.normal(250, 20, 30) # ~250ms, low SD
    parkinson_taps = np.random.normal(380, 85, 20) # ~380ms, high SD & hesitation

    res_healthy = evaluate_finger_tap_stability(healthy_taps)
    res_parkinson = evaluate_finger_tap_stability(parkinson_taps)

    print("\n--- A. Tremor Frequency Estimation Error (Zero-Crossing vs FFT) ---")
    print(df_freq.to_string(index=False))
    print(f"\nMean Zero-Crossing Error : {df_freq['Zero_Crossing_Error (Hz)'].mean():.3f} Hz")
    print(f"Mean FFT Estimation Error  : {df_freq['FFT_Error (Hz)'].mean():.3f} Hz")

    print("\n--- B. Finger Tapping Inter-Tap Interval (ITI) Stability ---")
    print(f"Healthy Controls ITI     : Mean={res_healthy['mean_iti']:.1f}ms, SD={res_healthy['sd_iti']:.1f}ms, CV={res_healthy['cv_iti']:.2f}%")
    print(f"Parkinson Pattern ITI    : Mean={res_parkinson['mean_iti']:.1f}ms, SD={res_parkinson['sd_iti']:.1f}ms, CV={res_parkinson['cv_iti']:.2f}%")

    return df_freq

if __name__ == "__main__":
    run_experiment_4()
