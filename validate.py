#!/usr/bin/env python3
"""
Φ-24 Temporal Resonator Validation Script
P-ECC Convergence and Riemann Lock Verification

Author: CTT Research Group
Version: 2.0 (24D Enhanced)
Date: 2026-02-09
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from scipy.stats import kstest
import h5py
import json
from typing import Dict, List, Tuple

class Phi24Validator:
    """Validate Φ-24 resonator performance metrics"""
    
    def __init__(self, config_file: str = None):
        """Initialize validator with configuration"""
        # Fundamental constants
        self.alpha_RH = 0.076587201  # Riemann Hypothesis condition
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.omega_0 = 587e3  # Base frequency (Hz)
        self.tau_w = 11e-9  # Temporal wedge (11 ns)
        
        # Performance thresholds (Grok-validated)
        self.thresholds = {
            'PECC_MIN': 0.985,  # Minimum P-ECC convergence score
            'GUE_R2_MIN': 0.99,  # Minimum GUE spacing correlation
            'FREQ_TOL': 100,  # Frequency tolerance (Hz)
            'PHASE_NOISE_MAX': -140,  # Maximum phase noise (dBc/Hz)
            'OFF_CRITICAL_MAX': 1e-6,  # Maximum off-critical power (V^2)
        }
        
        # 24 Riemann zeros (non-trivial, first 24)
        self.riemann_zeros = np.array([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425275
        ])
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.thresholds.update(config.get('thresholds', {}))
        print(f"Loaded configuration from {config_file}")
    
    def load_measurement_data(self, data_file: str) -> Dict:
        """Load measurement data from HDF5 file"""
        with h5py.File(data_file, 'r') as f:
            data = {
                'frequencies': f['frequencies'][:],
                'spectrum': f['spectrum'][:],
                'phase_noise': f['phase_noise'][:] if 'phase_noise' in f else None,
                'temp_wedge': f.attrs.get('temporal_wedge', 11e-9),
                'temperature': f.attrs.get('temperature', 0.02),
            }
        return data
    
    def calculate_PECC_score(self, frequencies: np.ndarray, 
                            powers: np.ndarray) -> float:
        """Calculate Prime-Specific Error Correction convergence score"""
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(powers, height=np.max(powers)*0.1, 
                                       distance=len(frequencies)//50)
        
        if len(peaks) < 24:
            print(f"Warning: Only {len(peaks)} peaks found, expected 24")
            return 0.0
        
        # Sort peaks by power
        peak_powers = powers[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1][:24]
        main_peaks = peaks[sorted_indices]
        
        # Calculate spacing between consecutive peaks
        peak_freqs = frequencies[main_peaks]
        spacings = np.diff(np.sort(peak_freqs))
        
        # Expected GUE (Gaussian Unitary Ensemble) spacing
        # Wigner surmise for GUE: P(s) = (32/π²) s² exp(-4s²/π)
        normalized_spacings = spacings / np.mean(spacings)
        
        # Generate theoretical GUE distribution
        s = np.linspace(0, 3, 1000)
        P_theory = (32/(np.pi**2)) * s**2 * np.exp(-4*s**2/np.pi)
        
        # Histogram of measured spacings
        hist, bin_edges = np.histogram(normalized_spacings, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate theoretical distribution at bin centers
        from scipy.interpolate import interp1d
        interp = interp1d(s, P_theory, bounds_error=False, fill_value=0)
        P_theory_interp = interp(bin_centers)
        
        # Calculate R² correlation
        mask = P_theory_interp > 0
        if np.sum(mask) < 10:
            return 0.0
        
        correlation = np.corrcoef(hist[mask], P_theory_interp[mask])[0,1]
        r_squared = correlation**2
        
        # Calculate P-ECC score (weighted combination)
        peak_alignment = self._check_peak_alignment(peak_freqs)
        spacing_consistency = 1.0 - np.std(normalized_spacings)/0.5
        
        # Final P-ECC score
        pecc_score = 0.4 * r_squared + 0.3 * peak_alignment + 0.3 * spacing_consistency
        
        return pecc_score
    
    def _check_peak_alignment(self, peak_freqs: np.ndarray) -> float:
        """Check alignment with Riemann zeros"""
        # Expected frequency scaling
        expected_ratios = self.riemann_zeros / self.riemann_zeros[0]
        
        # Normalize measured frequencies
        measured_ratios = peak_freqs / peak_freqs[0]
        
        # Calculate alignment score
        errors = np.abs(expected_ratios[:len(measured_ratios)] - measured_ratios)
        alignment_score = 1.0 - np.mean(errors) / 0.1  # Normalize by tolerance
        
        return max(0.0, min(1.0, alignment_score))
    
    def verify_riemann_lock(self, frequencies: np.ndarray, 
                           powers: np.ndarray) -> Dict:
        """Verify Riemann Lock condition"""
        
        # Find main resonance (should be at 1.485 MHz)
        main_idx = np.argmax(powers)
        main_freq = frequencies[main_idx]
        
        # Check if main resonance is at correct frequency
        target_freq = 1.485e6
        freq_error = np.abs(main_freq - target_freq)
        freq_ok = freq_error < self.thresholds['FREQ_TOL']
        
        # Calculate off-critical power (should be near zero)
        off_critical_mask = (np.abs(frequencies - target_freq) > 1e3)
        off_critical_power = np.mean(powers[off_critical_mask])
        off_critical_ok = off_critical_power < self.thresholds['OFF_CRITICAL_MAX']
        
        # Check for 24 distinct peaks
        peaks, _ = find_peaks(powers, height=np.max(powers)*0.01)
        num_peaks_ok = len(peaks) >= 24
        
        # Calculate P-ECC score
        pecc_score = self.calculate_PECC_score(frequencies, powers)
        pecc_ok = pecc_score >= self.thresholds['PECC_MIN']
        
        return {
            'riemann_lock': freq_ok and off_critical_ok and num_peaks_ok and pecc_ok,
            'main_frequency': main_freq,
            'frequency_error': freq_error,
            'off_critical_power': off_critical_power,
            'num_peaks': len(peaks),
            'pecc_score': pecc_score,
            'checks': {
                'frequency': freq_ok,
                'off_critical': off_critical_ok,
                'num_peaks': num_peaks_ok,
                'pecc_score': pecc_ok,
            }
        }
    
    def verify_temporal_wedge(self, time_data: np.ndarray,
                            signal_data: np.ndarray) -> Dict:
        """Verify 11 ns temporal wedge"""
        
        # Calculate autocorrelation to find coherence time
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find where autocorrelation drops to 1/e
        coherence_time_idx = np.where(autocorr < 1/np.e)[0]
        if len(coherence_time_idx) > 0:
            coherence_time = time_data[coherence_time_idx[0]]
        else:
            coherence_time = time_data[-1]
        
        # Check if coherence time matches 11 ns
        target_coherence = 11e-9
        coherence_error = np.abs(coherence_time - target_coherence)
        coherence_ok = coherence_error < 0.01e-9  # 10 ps tolerance
        
        # Calculate wedge sharpness (derivative at edge)
        derivative = np.gradient(autocorr, time_data)
        sharpness = np.max(np.abs(derivative))
        
        return {
            'coherence_time': coherence_time,
            'coherence_error': coherence_error,
            'coherence_ok': coherence_ok,
            'wedge_sharpness': sharpness,
            'autocorrelation': autocorr,
        }
    
    def verify_phase_noise(self, phase_noise_freq: np.ndarray,
                          phase_noise_power: np.ndarray) -> Dict:
        """Verify phase noise specifications"""
        
        # Check at 1 kHz offset
        idx_1khz = np.argmin(np.abs(phase_noise_freq - 1e3))
        phase_noise_1khz = phase_noise_power[idx_1khz]
        
        # Check at 10 kHz offset
        idx_10khz = np.argmin(np.abs(phase_noise_freq - 10e3))
        phase_noise_10khz = phase_noise_power[idx_10khz]
        
        phase_noise_ok = (phase_noise_1khz < self.thresholds['PHASE_NOISE_MAX'] and
                         phase_noise_10khz < self.thresholds['PHASE_NOISE_MAX'] - 10)
        
        return {
            'phase_noise_1khz': phase_noise_1khz,
            'phase_noise_10khz': phase_noise_10khz,
            'phase_noise_ok': phase_noise_ok,
        }
    
    def generate_report(self, validation_results: Dict) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("Φ-24 TEMPORAL RESONATOR VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {np.datetime64('now')}")
        report.append("")
        
        # Riemann Lock verification
        rl = validation_results['riemann_lock']
        report.append("RIEMANN LOCK VERIFICATION")
        report.append("-" * 40)
        report.append(f"Status: {'PASS' if rl['riemann_lock'] else 'FAIL'}")
        report.append(f"Main Frequency: {rl['main_frequency']/1e6:.6f} MHz")
        report.append(f"Frequency Error: {rl['frequency_error']:.1f} Hz")
        report.append(f"Off-Critical Power: {rl['off_critical_power']:.2e} V²")
        report.append(f"Number of Peaks: {rl['num_peaks']}")
        report.append(f"P-ECC Score: {rl['pecc_score']:.6f}")
        report.append(f"  Target: > {self.thresholds['PECC_MIN']}")
        report.append("")
        
        # Temporal wedge verification
        tw = validation_results['temporal_wedge']
        report.append("TEMPORAL WEDGE VERIFICATION")
        report.append("-" * 40)
        report.append(f"Coherence Time: {tw['coherence_time']/1e-9:.3f} ns")
        report.append(f"Target: 11.000 ns ± 0.010 ns")
        report.append(f"Error: {tw['coherence_error']/1e-12:.1f} ps")
        report.append(f"Status: {'PASS' if tw['coherence_ok'] else 'FAIL'}")
        report.append(f"Wedge Sharpness: {tw['wedge_sharpness']:.2e} s⁻¹")
        report.append("")
        
        # Phase noise verification
        if 'phase_noise' in validation_results:
            pn = validation_results['phase_noise']
            report.append("PHASE NOISE VERIFICATION")
            report.append("-" * 40)
            report.append(f"@ 1 kHz: {pn['phase_noise_1khz']:.1f} dBc/Hz")
            report.append(f"Target: < {self.thresholds['PHASE_NOISE_MAX']} dBc/Hz")
            report.append(f"@ 10 kHz: {pn['phase_noise_10khz']:.1f} dBc/Hz")
            report.append(f"Status: {'PASS' if pn['phase_noise_ok'] else 'FAIL'}")
            report.append("")
        
        # Overall status
        all_passed = (rl['riemann_lock'] and tw['coherence_ok'] and 
                     ('phase_noise' not in validation_results or 
                      pn['phase_noise_ok']))
        
        report.append("OVERALL VERIFICATION")
        report.append("-" * 40)
        report.append(f"STATUS: {'PASS - Φ-24 OPERATIONAL' if all_passed else 'FAIL - REQUIRES TUNING'}")
        
        if all_passed:
            report.append("")
            report.append("The resonator has achieved Riemann Lock with")
            report.append(f"P-ECC convergence score: {rl['pecc_score']:.6f}")
            report.append("Hardware is ready for LS-SAT computation.")
        else:
            report.append("")
            report.append("FAILURE ANALYSIS:")
            if not rl['riemann_lock']:
                report.append("- Riemann Lock not achieved")
                if not rl['checks']['frequency']:
                    report.append(f"  Frequency error: {rl['frequency_error']:.1f} Hz > {self.thresholds['FREQ_TOL']} Hz")
                if not rl['checks']['off_critical']:
                    report.append(f"  Off-critical power: {rl['off_critical_power']:.2e} > {self.thresholds['OFF_CRITICAL_MAX']}")
                if not rl['checks']['num_peaks']:
                    report.append(f"  Peaks found: {rl['num_peaks']} < 24")
                if not rl['checks']['pecc_score']:
                    report.append(f"  P-ECC score: {rl['pecc_score']:.6f} < {self.thresholds['PECC_MIN']}")
            if not tw['coherence_ok']:
                report.append(f"- Temporal wedge error: {tw['coherence_error']/1e-12:.1f} ps > 10 ps")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_results(self, frequencies: np.ndarray, powers: np.ndarray,
                    validation_results: Dict, save_path: str = None):
        """Generate validation plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Spectrum with Riemann zeros
        ax1 = axes[0, 0]
        ax1.plot(frequencies/1e6, 10*np.log10(powers + 1e-20))
        ax1.axvline(1.485, color='r', linestyle='--', alpha=0.5, 
                   label='Target: 1.485 MHz')
        
        # Mark Riemann zero positions
        zero_freqs = 1.485e6 * self.riemann_zeros / self.riemann_zeros[0]
        for freq in zero_freqs[:24]:
            ax1.axvline(freq/1e6, color='g', linestyle=':', alpha=0.3)
        
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (dB)')
        ax1.set_title('Resonance Spectrum with Riemann Zero Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GUE Spacing Distribution
        ax2 = axes[0, 1]
        peaks, _ = find_peaks(powers, height=np.max(powers)*0.1)
        if len(peaks) >= 24:
            peak_freqs = frequencies[peaks]
            peak_freqs = np.sort(peak_freqs)[:24]
            spacings = np.diff(peak_freqs)
            normalized_spacings = spacings / np.mean(spacings)
            
            hist, bin_edges = np.histogram(normalized_spacings, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Theoretical GUE distribution
            s = np.linspace(0, 3, 1000)
            P_theory = (32/(np.pi**2)) * s**2 * np.exp(-4*s**2/np.pi)
            
            ax2.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], 
                   alpha=0.5, label='Measured')
            ax2.plot(s, P_theory, 'r-', linewidth=2, label='GUE Theory')
            ax2.set_xlabel('Normalized Spacing')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('GUE Spacing Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temporal Wedge Autocorrelation
        ax3 = axes[1, 0]
        if 'temporal_wedge' in validation_results:
            tw = validation_results['temporal_wedge']
            if 'autocorrelation' in tw:
                time = np.linspace(0, 20e-9, len(tw['autocorrelation']))
                ax3.plot(time/1e-9, tw['autocorrelation'])
                ax3.axvline(11, color='r', linestyle='--', label='11 ns Target')
                ax3.axhline(1/np.e, color='g', linestyle=':', label='1/e')
                ax3.set_xlabel('Time (ns)')
                ax3.set_ylabel('Autocorrelation')
                ax3.set_title('Temporal Wedge Coherence')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Phase Noise
        ax4 = axes[1, 1]
        if 'phase_noise' in validation_results:
            pn = validation_results['phase_noise']
            if 'frequencies' in pn and 'spectrum' in pn:
                ax4.semilogx(pn['frequencies'], pn['spectrum'])
                ax4.axhline(self.thresholds['PHASE_NOISE_MAX'], color='r', 
                           linestyle='--', label='Spec Limit')
                ax4.set_xlabel('Offset Frequency (Hz)')
                ax4.set_ylabel('Phase Noise (dBc/Hz)')
                ax4.set_title('Phase Noise Spectrum')
                ax4.legend()
                ax4.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig

def main():
    """Main validation routine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Φ-24 Temporal Resonator Validator')
    parser.add_argument('data_file', help='HDF5 measurement data file')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--output', default='validation_report.txt', 
                       help='Output report file')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate validation plots')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = Phi24Validator(args.config)
    
    # Load measurement data
    try:
        data = validator.load_measurement_data(args.data_file)
        print(f"Loaded data from {args.data_file}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Perform validations
    validation_results = {}
    
    # Riemann Lock verification
    validation_results['riemann_lock'] = validator.verify_riemann_lock(
        data['frequencies'], data['spectrum'])
    
    # Temporal wedge verification (if time domain data available)
    if 'time_domain' in data:
        validation_results['temporal_wedge'] = validator.verify_temporal_wedge(
            data['time_domain']['time'],
            data['time_domain']['signal'])
    
    # Phase noise verification
    if data['phase_noise'] is not None:
        validation_results['phase_noise'] = validator.verify_phase_noise(
            data['phase_noise']['frequencies'],
            data['phase_noise']['spectrum'])
    
    # Generate report
    report = validator.generate_report(validation_results)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    print(f"Report saved to {args.output}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    rl = validation_results['riemann_lock']
    print(f"Riemann Lock: {'PASS' if rl['riemann_lock'] else 'FAIL'}")
    print(f"P-ECC Score: {rl['pecc_score']:.6f}")
    if 'temporal_wedge' in validation_results:
        tw = validation_results['temporal_wedge']
        print(f"Temporal Wedge: {tw['coherence_time']/1e-9:.3f} ns")
    
    # Generate plots if requested
    if args.plot:
        plot_file = args.output.replace('.txt', '.png')
        validator.plot_results(data['frequencies'], data['spectrum'],
                              validation_results, plot_file)
    
    return 0

if __name__ == "__main__":
    exit(main())
