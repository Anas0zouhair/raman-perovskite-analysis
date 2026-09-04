import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve
from lmfit.models import VoigtModel, LorentzianModel, GaussianModel, BreitWignerModel
import json
from pathlib import Path

# ==========================================
# CONFIGURATION FOR HALIDE PEROVSKITES
# ==========================================

class RamanConfig:
    """Configuration for Raman spectroscopy analysis"""
    
    def __init__(self, config_file=None):
        self.laser_wavelength_nm = 532.0
        self.cosmic_ray_threshold = 5
        self.airpls_lambda = 1e6
        self.savgol_window = 11
        self.savgol_polyorder = 2
        self.peaks = []
        
        if config_file:
            self.load_from_json(config_file)
    
    def load_from_json(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.__dict__.update(config)
    
    def save_to_json(self, output_file):
        """Save configuration to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def add_peak(self, name, center, amplitude, sigma, center_min=None, center_max=None, model_type='Voigt'):
        """Add a peak to fit configuration"""
        peak = {
            'name': name,
            'center': center,
            'amplitude': amplitude,
            'sigma': sigma,
            'center_min': center_min,
            'center_max': center_max,
            'model_type': model_type
        }
        self.peaks.append(peak)

# ==========================================
# 1. SIGNAL PREPROCESSING FUNCTIONS
# ==========================================

def remove_cosmic_rays(y, threshold=5):
    """
    Détecte et supprime les pics cosmiques (spikes parasites très étroits).
    Compare la dérivée locale à l'écart-type général.
    
    Parameters
    ----------
    y : np.ndarray
        Signal intensity array
    threshold : float
        Multiplier for standard deviation threshold
    
    Returns
    -------
    np.ndarray
        Cleaned signal
    """
    y_clean = y.copy()
    dy = np.abs(np.diff(y_clean))
    spike_indices = np.where(dy > threshold * np.std(dy))[0] + 1
    for idx in spike_indices:
        if 0 < idx < len(y_clean) - 1:
            y_clean[idx] = (y_clean[idx - 1] + y_clean[idx + 1]) / 2.0
    return y_clean

def airpls_baseline(y, lambda_=1e6, porder=1, max_iter=50):
    """
    Soustrait la ligne de base (fluorescence) via l'algorithme airPLS.
    Adaptive Iteratively Reweighted Penalized Least Squares
    
    Parameters
    ----------
    y : np.ndarray
        Signal intensity
    lambda_ : float
        Smoothing parameter (1e4 to 1e7, higher = smoother)
    porder : int
        Polynomial order
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    np.ndarray
        Estimated baseline
    """
    m = y.shape[0]
    D = np.diff(eye(m).toarray(), n=2, axis=0)
    D = csc_matrix(D)
    
    w = np.ones(m)
    for i in range(1, max_iter):
        W = csc_matrix((w, (np.arange(m), np.arange(m))), shape=(m, m))
        Z = W + lambda_ * D.T.dot(D)
        z = spsolve(Z, w * y)
        d = y - z
        
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * np.abs(y).sum() or i == max_iter - 1:
            break
            
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * np.abs(d[0]) / dssn)
        w[-1] = np.exp(i * np.abs(d[-1]) / dssn)
        
    return z

# ==========================================
# 2. DATA LOADING AND CONVERSION
# ==========================================

def load_spectrum(filename, laser_wavelength_nm):
    """
    Load experimental spectrum and convert wavelength to Raman shift
    
    Parameters
    ----------
    filename : str
        Path to spectrum file (2 columns: wavelength_nm, intensity)
    laser_wavelength_nm : float
        Laser excitation wavelength in nm
    
    Returns
    -------
    x : np.ndarray
        Raman shift (cm^-1)
    y : np.ndarray
        Intensity
    """
    data = np.loadtxt(filename, delimiter="\t")
    x_nm = data[:, 0]
    y = data[:, 1]
    
    # Convert nm to Raman shift (cm^-1)
    x = ((1.0 / laser_wavelength_nm) - (1.0 / x_nm)) * 1e7
    
    # Sort by Raman shift
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    y = y[sort_indices]
    
    return x, y

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================

def preprocess_spectrum(y, config):
    """
    Apply complete preprocessing pipeline:
    cosmic ray removal → baseline subtraction → smoothing
    
    Parameters
    ----------
    y : np.ndarray
        Raw intensity
    config : RamanConfig
        Configuration object
    
    Returns
    -------
    y_processed : np.ndarray
        Preprocessed spectrum
    baseline : np.ndarray
        Estimated baseline
    """
    # Remove cosmic rays
    y_despiked = remove_cosmic_rays(y, threshold=config.cosmic_ray_threshold)
    
    # Subtract baseline
    baseline = airpls_baseline(y_despiked, lambda_=config.airpls_lambda)
    y_baselined = y_despiked - baseline
    
    # Savitzky-Golay smoothing
    y_smooth = savgol_filter(y_baselined, window_length=config.savgol_window, 
                             polyorder=config.savgol_polyorder)
    
    return y_smooth, baseline

# ==========================================
# 4. GENERALIZED MULTI-PEAK FITTING
# ==========================================

def build_model_from_config(config):
    """
    Build composite model from peak configuration
    
    Parameters
    ----------
    config : RamanConfig
        Configuration with peaks
    
    Returns
    -------
    model : lmfit Model
        Composite model
    pars : Parameters
        Initial parameters
    """
    model_dict = {
        'Voigt': VoigtModel,
        'Lorentzian': LorentzianModel,
        'Gaussian': GaussianModel,
        'BreitWigner': BreitWignerModel
    }
    
    model = None
    pars = None
    
    for peak_config in config.peaks:
        prefix = f"{peak_config['name']}_"
        model_class = model_dict.get(peak_config['model_type'], VoigtModel)
        peak_model = model_class(prefix=prefix)
        
        # Add to composite model
        if model is None:
            model = peak_model
        else:
            model = model + peak_model
        
        # Initialize parameters
        peak_pars = peak_model.make_params(
            center=peak_config['center'],
            amplitude=peak_config['amplitude'],
            sigma=peak_config['sigma']
        )
        
        # Apply constraints
        if peak_config['center_min'] is not None:
            peak_pars[f'{prefix}center'].set(min=peak_config['center_min'])
        if peak_config['center_max'] is not None:
            peak_pars[f'{prefix}center'].set(max=peak_config['center_max'])
        
        if pars is None:
            pars = peak_pars
        else:
            pars.update(peak_pars)
    
    return model, pars

def fit_spectrum(x, y, model, pars):
    """
    Perform non-linear least-squares fitting
    
    Parameters
    ----------
    x : np.ndarray
        Raman shift
    y : np.ndarray
        Intensity
    model : lmfit Model
        Composite model
    pars : Parameters
        Initial parameters
    
    Returns
    -------
    result : ModelResult
        Fitting result
    """
    result = model.fit(y, pars, x=x)
    return result

# ==========================================
# 5. RESULTS EXTRACTION AND EXPORT
# ==========================================

def extract_peak_metrics(result, config):
    """
    Extract peak metrics from fit result
    
    Parameters
    ----------
    result : ModelResult
        Fitting result
    config : RamanConfig
        Configuration
    
    Returns
    -------
    df : pd.DataFrame
        Peak metrics table
    """
    fit_data = []
    
    for peak_config in config.peaks:
        prefix = f"{peak_config['name']}_"
        try:
            fit_data.append({
                'Peak': peak_config['name'],
                'Model': peak_config['model_type'],
                'Position (cm⁻¹)': result.params[f'{prefix}center'].value,
                'Err Position': result.params[f'{prefix}center'].stderr,
                'FWHM (cm⁻¹)': result.params[f'{prefix}fwhm'].value if f'{prefix}fwhm' in result.params else None,
                'Amplitude': result.params[f'{prefix}amplitude'].value,
                'Sigma': result.params[f'{prefix}sigma'].value,
                'Height': result.params[f'{prefix}height'].value if f'{prefix}height' in result.params else None,
            })
        except KeyError:
            print(f"Warning: Could not extract metrics for {peak_config['name']}")
    
    return pd.DataFrame(fit_data)

def export_results(x, y_raw, y_smooth, result, df_metrics, output_prefix="raman_output"):
    """
    Export all results to CSV files
    
    Parameters
    ----------
    x : np.ndarray
        Raman shift
    y_raw : np.ndarray
        Raw intensity
    y_smooth : np.ndarray
        Preprocessed intensity
    result : ModelResult
        Fitting result
    df_metrics : pd.DataFrame
        Peak metrics
    output_prefix : str
        Output file prefix
    """
    # Export peak metrics
    metrics_file = f"{output_prefix}_peak_metrics.csv"
    df_metrics.to_csv(metrics_file, index=False)
    print(f"[INFO] Peak metrics exported to '{metrics_file}'")
    
    # Export spectrum data
    spectrum_file = f"{output_prefix}_spectrum.csv"
    export_spectrum = pd.DataFrame({
        'Raman_Shift_cm1': x,
        'Intensity_Raw': y_raw,
        'Intensity_Processed': y_smooth,
        'Fit_Total': result.best_fit
    })
    export_spectrum.to_csv(spectrum_file, index=False)
    print(f"[INFO] Spectrum data exported to '{spectrum_file}'")
    
    # Export fit report
    report_file = f"{output_prefix}_fit_report.txt"
    with open(report_file, 'w') as f:
        f.write(result.fit_report())
    print(f"[INFO] Fit report exported to '{report_file}'")

# ==========================================
# 6. PUBLICATION-QUALITY PLOTTING
# ==========================================

def plot_spectrum(x, y_smooth, result, config, output_prefix="raman_output"):
    """
    Generate publication-quality plot
    
    Parameters
    ----------
    x : np.ndarray
        Raman shift
    y_smooth : np.ndarray
        Preprocessed intensity
    result : ModelResult
        Fitting result
    config : RamanConfig
        Configuration
    output_prefix : str
        Output file prefix
    """
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 11
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Experimental data
    ax.plot(x, y_smooth, 'ko', markersize=2, alpha=0.5, label='Experimental data')
    
    # Total fit
    ax.plot(x, result.best_fit, 'r-', lw=2.5, label='Global fit')
    
    # Individual components
    components = result.eval_components(x=x)
    colors = plt.cm.tab10(np.linspace(0, 1, len(config.peaks)))
    
    for i, peak_config in enumerate(config.peaks):
        prefix = f"{peak_config['name']}_"
        if prefix in components:
            ax.plot(x, components[prefix], '--', color=colors[i], 
                   lw=1.5, label=peak_config['name'])
    
    ax.set_xlabel(r'Raman Shift ($\mathrm{cm^{-1}}$)', fontsize=12)
    ax.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax.legend(frameon=True, edgecolor='black', loc='best')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    
    # Save files
    png_file = f"{output_prefix}.png"
    pdf_file = f"{output_prefix}.pdf"
    
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"[INFO] Figures saved: '{png_file}' (PNG) and '{pdf_file}' (PDF)")
    
    plt.close(fig)

# ==========================================
# 7. MAIN EXECUTION FUNCTION
# ==========================================

def analyze_raman_spectrum(spectrum_file, config_file, output_prefix="raman_output"):
    """
    Complete Raman spectroscopy analysis pipeline
    
    Parameters
    ----------
    spectrum_file : str
        Path to spectrum data file
    config_file : str
        Path to configuration JSON file
    output_prefix : str
        Output file prefix
    """
    print("="*60)
    print("RAMAN SPECTROSCOPY ANALYSIS PIPELINE")
    print("="*60)
    
    # Load configuration
    config = RamanConfig(config_file)
    print(f"\n[1] Configuration loaded from '{config_file}'")
    print(f"    Laser wavelength: {config.laser_wavelength_nm} nm")
    print(f"    Number of peaks: {len(config.peaks)}")
    
    # Load spectrum
    x, y_raw = load_spectrum(spectrum_file, config.laser_wavelength_nm)
    print(f"\n[2] Spectrum loaded from '{spectrum_file}'")
    print(f"    Data points: {len(x)}")
    print(f"    Raman shift range: {x.min():.1f} - {x.max():.1f} cm⁻¹")
    
    # Preprocessing
    y_smooth, baseline = preprocess_spectrum(y_raw, config)
    print(f"\n[3] Preprocessing complete")
    print(f"    - Cosmic ray removal: threshold = {config.cosmic_ray_threshold}σ")
    print(f"    - Baseline subtraction: airPLS (λ={config.airpls_lambda:.0e})")
    print(f"    - Smoothing: Savitzky-Golay (window={config.savgol_window}, order={config.savgol_polyorder})")
    
    # Build and fit model
    model, pars = build_model_from_config(config)
    print(f"\n[4] Model built with {len(config.peaks)} peaks")
    
    result = fit_spectrum(x, y_smooth, model, pars)
    print(f"\n[5] Fitting complete")
    print(f"    Reduced χ²: {result.redchi:.4f}")
    print(f"    R-squared: {result.rsquared:.4f}")
    
    # Extract metrics
    df_metrics = extract_peak_metrics(result, config)
    print(f"\n[6] Peak metrics extracted")
    print(df_metrics.to_string(index=False))
    
    # Export results
    export_results(x, y_raw, y_smooth, result, df_metrics, output_prefix)
    
    # Plot
    plot_spectrum(x, y_smooth, result, config, output_prefix)
    
    print(f"\n[7] Analysis complete!")
    print(f"    Output prefix: '{output_prefix}'")
    print("="*60)
    
    return result, df_metrics

if __name__ == "__main__":
    # Example usage
    analyze_raman_spectrum(
        spectrum_file="vrai_spectre.txt",
        config_file="raman_config.json",
        output_prefix="raman_output"
    )
