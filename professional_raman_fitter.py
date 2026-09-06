"""
PROFESSIONAL RAMAN SPECTROSCOPY PEAK FITTING TOOL
================================================

Based on best practices from peer-reviewed literature:
- Voigt/Lorentzian/Gaussian peak models (user choice)
- airPLS baseline correction (Zhang et al. 2010)
- Levenberg-Marquardt optimization
- Full user control at every step
- Expert diagnostics and validation

This is a PRODUCTION-READY tool for research-quality analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, sparse, optimize
from scipy.sparse.linalg import spsolve
from lmfit import Model, Parameters, Minimizer
from lmfit.models import (VoigtModel, LorentzianModel, GaussianModel, 
                          ConstantModel, LinearModel)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class RamanPeakFitter:
    """
    Professional Raman spectroscopy analysis with full user control
    """
    
    def __init__(self):
        self.x = None
        self.y = None
        self.y_baseline_corrected = None
        self.baseline = None
        self.params_dict = {}
        self.fit_result = None
        self.config = {}
        
    # ================================================================
    # STEP 1: DATA LOADING
    # ================================================================
    
    def load_data(self):
        """Load spectrum from file with full control"""
        print("\n" + "="*70)
        print("STEP 1: LOAD YOUR SPECTRUM")
        print("="*70)
        
        while True:
            filepath = input("\n📁 Enter spectrum file path (.txt or .csv): ").strip()
            
            try:
                if filepath.endswith('.csv'):
                    delimiter = ','
                else:
                    delimiter = '\t'
                
                data = np.loadtxt(filepath, delimiter=delimiter, skiprows=1)
                
                if data.shape[1] < 2:
                    print(f"❌ Need 2 columns, got {data.shape[1]}")
                    continue
                
                self.x = data[:, 0]
                self.y = data[:, 1]
                
                print(f"\n✅ Loaded successfully!")
                print(f"   File: {filepath}")
                print(f"   Data points: {len(self.x)}")
                print(f"   X range: {self.x.min():.2f} - {self.x.max():.2f}")
                print(f"   Y range: {self.y.min():.2f} - {self.y.max():.2f}")
                
                return True
                
            except FileNotFoundError:
                print(f"❌ File not found: {filepath}")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # ================================================================
    # STEP 2: DATA PREPROCESSING
    # ================================================================
    
    def airpls_baseline(self, lambda_=1e6, porder=1, itermax=15):
        """
        Adaptive Iteratively Reweighted Penalized Least Squares (airPLS)
        Reference: Zhang et al. 2010, Analyst 135:1138-1146
        
        This is the GOLD STANDARD for Raman baseline correction.
        """
        print("\n" + "─"*70)
        print("Applying airPLS baseline subtraction...")
        print(f"   λ={lambda_:.0e}, polynomial order={porder}")
        
        m = len(self.y)
        D = np.diff(np.eye(m), n=2, axis=0)
        D = sparse.csc_matrix(D)
        
        w = np.ones(m)
        for i in range(itermax):
            W = sparse.diags(w, 0)
            Z = W + lambda_ * D.T.dot(D)
            z = spsolve(Z, w * self.y)
            
            d = self.y - z
            dssn = np.abs(d[d < 0].sum())
            
            if dssn < 0.001 * np.abs(self.y).sum() or i > 10:
                break
            
            w[d >= 0] = 0
            w[d < 0] = np.exp(i * np.abs(d[d < 0]) / (dssn + 1e-10))
            w[0] = np.exp(i * np.abs(d[0]) / (dssn + 1e-10))
            w[-1] = np.exp(i * np.abs(d[-1]) / (dssn + 1e-10))
        
        self.baseline = z
        self.y_baseline_corrected = self.y - self.baseline
        print(f"✅ Baseline corrected")
        return self.y_baseline_corrected
    
    def preprocess_interactive(self):
        """Interactive preprocessing with full control"""
        print("\n" + "="*70)
        print("STEP 2: PREPROCESSING")
        print("="*70)
        
        print("\n🔹 OPTION 1: Baseline Subtraction Method")
        print("   1) airPLS (RECOMMENDED - research-grade)")
        print("   2) Polynomial fit")
        print("   3) None (use raw data)")
        
        while True:
            choice = input("\n   Select (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("   ❌ Invalid choice")
        
        if choice == '1':
            print("\n   airPLS PARAMETERS:")
            print("   - λ (lambda): controls smoothness")
            print("     λ=1e4: More baseline structure")
            print("     λ=1e6: Balanced (RECOMMENDED)")
            print("     λ=1e8: Very smooth baseline")
            
            while True:
                try:
                    lambda_str = input("   Enter λ (default=1e6): ").strip()
                    lambda_val = float(lambda_str) if lambda_str else 1e6
                    if 1e3 <= lambda_val <= 1e10:
                        break
                    print("   ❌ Enter value between 1e3 and 1e10")
                except:
                    print("   ❌ Invalid input")
            
            self.airpls_baseline(lambda_=lambda_val)
            self.config['baseline_method'] = 'airPLS'
            self.config['airpls_lambda'] = lambda_val
            
        elif choice == '2':
            print("\n   Polynomial order (typically 2-5 for Raman)")
            while True:
                try:
                    poly_order = int(input("   Enter polynomial order (default=3): ") or "3")
                    if 1 <= poly_order <= 10:
                        break
                    print("   ❌ Enter order between 1-10")
                except:
                    print("   ❌ Invalid input")
            
            coeffs = np.polyfit(self.x, self.y, poly_order)
            self.baseline = np.polyval(coeffs, self.x)
            self.y_baseline_corrected = self.y - self.baseline
            self.config['baseline_method'] = 'polynomial'
            self.config['poly_order'] = poly_order
            print(f"   ✅ Polynomial baseline fitted (order={poly_order})")
            
        else:
            self.y_baseline_corrected = self.y.copy()
            self.baseline = np.zeros_like(self.y)
            self.config['baseline_method'] = 'none'
            print("   ✅ No baseline correction")
        
        # Smoothing
        print("\n🔹 OPTION 2: Smoothing (optional)")
        smooth = input("   Apply Savitzky-Golay smoothing? (y/n) [default=n]: ").strip().lower()
        
        if smooth == 'y':
            print("   Window length: typically 5-15 (odd number)")
            while True:
                try:
                    window = int(input("   Enter window length (default=11): ") or "11")
                    if window % 2 == 1 and window > 0:
                        break
                    print("   ❌ Enter odd number > 0")
                except:
                    print("   ❌ Invalid input")
            
            self.y_baseline_corrected = signal.savgol_filter(
                self.y_baseline_corrected, 
                window_length=window, 
                polyorder=2
            )
            self.config['smoothing'] = True
            self.config['savgol_window'] = window
            print(f"   ✅ Smoothed with window={window}")
        else:
            self.config['smoothing'] = False
        
        # Ensure non-negative
        self.y_baseline_corrected = np.maximum(self.y_baseline_corrected, 0)
    
    # ================================================================
    # STEP 3: MANUAL PEAK SPECIFICATION
    # ================================================================
    
    def specify_peaks_interactive(self):
        """User specifies each peak manually with full control"""
        print("\n" + "="*70)
        print("STEP 3: SPECIFY PEAKS MANUALLY")
        print("="*70)
        print("\nYou will define each peak's approximate position and initial guess.")
        print("The fitter will then refine these parameters.\n")
        
        peaks = []
        peak_num = 1
        
        while True:
            print(f"\n{'─'*70}")
            print(f"PEAK {peak_num}")
            print(f"{'─'*70}")
            
            # Center position
            while True:
                try:
                    center = float(input(f"Center position (cm⁻¹) [peak {peak_num}]: "))
                    break
                except:
                    print("❌ Enter a number")
            
            # Initial amplitude
            while True:
                try:
                    amplitude = float(input(f"Initial amplitude/height [peak {peak_num}]: "))
                    if amplitude > 0:
                        break
                    print("❌ Enter positive number")
                except:
                    print("❌ Enter a number")
            
            # Width (sigma/FWHM)
            print(f"\nWidth options for peak {peak_num}:")
            print("- Sigma: standard deviation (for Lorentzian/Gaussian)")
            print("- FWHM: Full Width at Half Maximum (common in spectroscopy)")
            print("Typical values: 5-50 cm⁻¹ depending on material and resolution")
            
            while True:
                try:
                    sigma = float(input(f"Width (sigma, cm⁻¹) [peak {peak_num}]: "))
                    if sigma > 0:
                        break
                    print("❌ Enter positive number")
                except:
                    print("❌ Enter a number")
            
            # Bounds
            print(f"\nParameter bounds for peak {peak_num}:")
            print("(These constrain the fit to physically reasonable values)")
            
            while True:
                try:
                    center_min = float(input(f"   Center min (cm⁻¹) [default={center-50}]: ") or str(center-50))
                    center_max = float(input(f"   Center max (cm⁻¹) [default={center+50}]: ") or str(center+50))
                    if center_min < center_max:
                        break
                    print("❌ Min must be less than max")
                except:
                    print("❌ Enter numbers")
            
            peaks.append({
                'num': peak_num,
                'center': center,
                'amplitude': amplitude,
                'sigma': sigma,
                'center_min': center_min,
                'center_max': center_max
            })
            
            print(f"\n✅ Peak {peak_num} defined:")
            print(f"   Center: {center:.1f} cm⁻¹")
            print(f"   Amplitude: {amplitude:.1f}")
            print(f"   Width (σ): {sigma:.1f} cm⁻¹")
            print(f"   Bounds: [{center_min:.1f}, {center_max:.1f}]")
            
            # Add more?
            add_more = input(f"\nAdd another peak? (y/n): ").strip().lower()
            if add_more != 'y':
                break
            
            peak_num += 1
        
        self.config['peaks'] = peaks
        print(f"\n✅ Total peaks defined: {len(peaks)}")
        return peaks
    
    # ================================================================
    # STEP 4: BACKGROUND MODEL
    # ================================================================
    
    def choose_background_model(self):
        """Choose background/baseline model"""
        print("\n" + "="*70)
        print("STEP 4: BACKGROUND MODEL")
        print("="*70)
        print("\nYour spectrum may have residual background after baseline subtraction.")
        print("Choose how to model it:\n")
        
        print("1) CONSTANT background (Recommended for most Raman)")
        print("   - Simple, robust")
        print("   - Good for already baseline-corrected data")
        print("\n2) LINEAR background")
        print("   - Handles slight slope")
        print("\n3) NONE (assume background = 0)")
        print("   - Best if baseline already subtracted well")
        
        while True:
            choice = input("\nSelect background model (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("❌ Invalid choice")
        
        if choice == '1':
            self.config['background_model'] = 'constant'
            print("✅ Background: CONSTANT")
        elif choice == '2':
            self.config['background_model'] = 'linear'
            print("✅ Background: LINEAR")
        else:
            self.config['background_model'] = 'none'
            print("✅ Background: NONE")
    
    # ================================================================
    # STEP 5: PEAK MODEL SELECTION
    # ================================================================
    
    def choose_peak_models(self, num_peaks):
        """Choose peak model for each peak"""
        print("\n" + "="*70)
        print("STEP 5: SELECT PEAK MODELS")
        print("="*70)
        print("\nChoose peak shape function for each peak.\n")
        
        print("📚 THEORY:")
        print("━" * 70)
        print("VOIGT (RECOMMENDED):")
        print("  • Convolution of Lorentzian + Gaussian")
        print("  • Best for most experimental Raman data")
        print("  • Accounts for both instrument (Gaussian) & sample (Lorentzian)")
        print("  • More parameters to fit\n")
        
        print("LORENTZIAN:")
        print("  • Natural lifetime broadening")
        print("  • Good for well-crystallized, ordered materials")
        print("  • Sharper peaks with longer tails\n")
        
        print("GAUSSIAN:")
        print("  • Inhomogeneous broadening")
        print("  • Good for amorphous/disordered materials")
        print("  • Broader, smoother tails\n")
        
        models = {}
        for i in range(num_peaks):
            print(f"Peak {i+1}:")
            print("  1) VOIGT (default)")
            print("  2) LORENTZIAN")
            print("  3) GAUSSIAN")
            
            while True:
                choice = input(f"  Select for peak {i+1} (1/2/3) [default=1]: ").strip() or "1"
                if choice in ['1', '2', '3']:
                    break
                print("  ❌ Invalid choice")
            
            if choice == '2':
                models[i] = 'lorentzian'
                print(f"  ✅ Peak {i+1}: LORENTZIAN\n")
            elif choice == '3':
                models[i] = 'gaussian'
                print(f"  ✅ Peak {i+1}: GAUSSIAN\n")
            else:
                models[i] = 'voigt'
                print(f"  ✅ Peak {i+1}: VOIGT\n")
        
        self.config['peak_models'] = models
        return models
    
    # ================================================================
    # STEP 6: BUILD FITTING MODEL
    # ================================================================
    
    def build_model(self, peaks, peak_models):
        """Build composite lmfit model"""
        print("\n" + "─"*70)
        print("Building composite model...")
        
        # Background
        if self.config['background_model'] == 'constant':
            model = ConstantModel()
            params = model.make_params(c=0)
            params['c'].set(min=0)
        elif self.config['background_model'] == 'linear':
            model = LinearModel()
            params = model.make_params(slope=0, intercept=0)
            params['intercept'].set(min=0)
        else:
            model = None
            params = Parameters()
        
        # Peaks
        for i, peak in enumerate(peaks):
            peak_model = peak_models.get(i, 'voigt')
            prefix = f"p{i+1}_"
            
            if peak_model == 'lorentzian':
                p_model = LorentzianModel(prefix=prefix)
            elif peak_model == 'gaussian':
                p_model = GaussianModel(prefix=prefix)
            else:  # voigt
                p_model = VoigtModel(prefix=prefix)
            
            p_params = p_model.make_params(
                center=peak['center'],
                amplitude=peak['amplitude'],
                sigma=peak['sigma']
            )
            
            # Set constraints
            p_params[f'{prefix}center'].set(
                min=peak['center_min'],
                max=peak['center_max']
            )
            p_params[f'{prefix}sigma'].set(min=0.1, max=500)
            p_params[f'{prefix}amplitude'].set(min=0)  # NO NEGATIVE PEAKS!
            
            if model is None:
                model = p_model
            else:
                model = model + p_model
            
            params.update(p_params)
        
        print(f"✅ Model built with {len(peaks)} peaks")
        print(f"   Total parameters: {len(params)}")
        
        return model, params
    
    # ================================================================
    # STEP 7: FITTING CONFIGURATION
    # ================================================================
    
    def configure_fitting(self):
        """Configure fitting algorithm parameters"""
        print("\n" + "="*70)
        print("STEP 6: FITTING CONFIGURATION")
        print("="*70)
        
        print("\nFitting Algorithm: Levenberg-Marquardt (industry standard)\n")
        
        print("Maximum iterations:")
        print("  Default: 5000 (usually sufficient)")
        print("  Large/complex: 10000-20000")
        
        while True:
            try:
                max_iter = int(input("Enter max iterations [default=5000]: ") or "5000")
                if max_iter > 0:
                    break
                print("❌ Enter positive number")
            except:
                print("❌ Invalid input")
        
        self.config['max_iter'] = max_iter
        
        print(f"\n✅ Fitting will use up to {max_iter} iterations")
    
    # ================================================================
    # STEP 8: PERFORM FITTING
    # ================================================================
    
    def fit(self, model, params):
        """Perform the fit with diagnostics"""
        print("\n" + "="*70)
        print("STEP 7: PERFORMING FIT")
        print("="*70)
        print("\nFitting data using Levenberg-Marquardt algorithm...")
        print("This may take 10-30 seconds depending on data size...\n")
        
        try:
            self.fit_result = model.fit(
                self.y_baseline_corrected,
                params,
                x=self.x,
                max_nfev=self.config.get('max_iter', 5000)
            )
            
            print(f"✅ Fitting complete!\n")
            print(f"   Reduced χ² = {self.fit_result.redchi:.6f}")
            print(f"   R-squared  = {self.fit_result.rsquared:.6f}")
            print(f"   Iterations = {self.fit_result.nfev}")
            print(f"   Status: {self.fit_result.message}\n")
            
            # Residuals analysis
            residuals = self.y_baseline_corrected - self.fit_result.best_fit
            rmse = np.sqrt(np.mean(residuals**2))
            print(f"   RMSE = {rmse:.6f}")
            
            if self.fit_result.redchi < 1:
                print("\n   ✅ EXCELLENT FIT (χ² << 1)")
            elif self.fit_result.redchi < 5:
                print("\n   ✅ GOOD FIT (χ² < 5)")
            elif self.fit_result.redchi < 100:
                print("\n   ⚠️  ACCEPTABLE FIT (χ² < 100)")
                print("      Consider checking peak initial guesses")
            else:
                print("\n   ❌ POOR FIT (χ² > 100)")
                print("      Try adjusting peak positions or model types")
            
            return True
            
        except Exception as e:
            print(f"❌ Fitting failed: {e}")
            return False
    
    # ================================================================
    # STEP 9: RESULTS & VISUALIZATION
    # ================================================================
    
    def print_results(self):
        """Print detailed results"""
        print("\n" + "="*70)
        print("STEP 8: FITTING RESULTS")
        print("="*70 + "\n")
        
        print(self.fit_result.fit_report())
        
        # Extract peak metrics
        print("\n" + "="*70)
        print("PEAK PARAMETERS SUMMARY")
        print("="*70 + "\n")
        
        results_data = []
        for key, param in self.fit_result.params.items():
            if 'p' in key and 'center' in key:
                peak_num = key.split('_')[0]
                center = param.value
                center_err = param.stderr or 0
                
                sigma_key = key.replace('center', 'sigma')
                sigma = self.fit_result.params[sigma_key].value
                
                amplitude_key = key.replace('center', 'amplitude')
                amplitude = self.fit_result.params[amplitude_key].value
                
                fwhm = 2.355 * sigma if 'sigma' in self.fit_result.params else None
                
                results_data.append({
                    'Peak': peak_num,
                    'Center (cm⁻¹)': f"{center:.2f}±{center_err:.4f}",
                    'FWHM (cm⁻¹)': f"{fwhm:.2f}" if fwhm else "N/A",
                    'Amplitude': f"{amplitude:.2f}",
                })
        
        df = pd.DataFrame(results_data)
        print(df.to_string(index=False))
    
    def plot_results(self, output_prefix='raman_fit'):
        """Create publication-quality plots"""
        print("\n" + "="*70)
        print("STEP 9: GENERATING PLOTS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Main fit
        ax = axes[0, 0]
        ax.plot(self.x, self.y_baseline_corrected, 'ko', markersize=4, 
                alpha=0.6, label='Data')
        ax.plot(self.x, self.fit_result.best_fit, 'r-', lw=2.5, 
                label='Best Fit')
        ax.fill_between(self.x, 0, self.fit_result.best_fit, 
                        alpha=0.2, color='red')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_title('Global Fit', fontweight='bold', fontsize=12)
        
        # Plot 2: Components
        ax = axes[0, 1]
        ax.plot(self.x, self.y_baseline_corrected, 'k-', lw=1, 
                alpha=0.5, label='Data')
        
        components = self.fit_result.eval_components(x=self.x)
        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
        
        for i, (name, comp) in enumerate(components.items()):
            if name != 'constant' and name != 'linear':
                ax.plot(self.x, comp, '--', color=colors[i], lw=2, label=name)
        
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_title('Peak Components', fontweight='bold', fontsize=12)
        
        # Plot 3: Residuals
        ax = axes[1, 0]
        residuals = self.y_baseline_corrected - self.fit_result.best_fit
        ax.plot(self.x, residuals, 'b-', lw=1.5, label='Residuals')
        ax.axhline(y=0, color='r', linestyle='--', lw=1)
        ax.fill_between(self.x, 0, residuals, where=(residuals>=0), 
                        alpha=0.3, color='green')
        ax.fill_between(self.x, 0, residuals, where=(residuals<0), 
                        alpha=0.3, color='red')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Residual (a.u.)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Residuals (χ²={self.fit_result.redchi:.4f})', 
                    fontweight='bold', fontsize=12)
        
        # Plot 4: Baseline
        ax = axes[1, 1]
        ax.plot(self.x, self.y, 'k-', lw=1, alpha=0.6, label='Raw spectrum')
        ax.plot(self.x, self.baseline, 'b-', lw=2, label='Baseline')
        ax.fill_between(self.x, 0, self.baseline, alpha=0.2, color='blue')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3)
        ax.set_title('Baseline Subtraction', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(f'{output_prefix}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_prefix}.pdf', bbox_inches='tight')
        print(f"✅ Plots saved:")
        print(f"   {output_prefix}.png (300 DPI)")
        print(f"   {output_prefix}.pdf (vector)")
        
        plt.show()
    
    def export_results(self, output_prefix='raman_fit'):
        """Export all results to CSV"""
        print("\n" + "─"*70)
        print("Exporting results to CSV...\n")
        
        # Peak metrics
        peak_data = []
        for key, param in self.fit_result.params.items():
            if 'p' in key and 'center' in key:
                peak_num = key.split('_')[0]
                center = param.value
                center_err = param.stderr or 0
                
                sigma_key = key.replace('center', 'sigma')
                sigma = self.fit_result.params[sigma_key].value
                
                amplitude_key = key.replace('center', 'amplitude')
                amplitude = self.fit_result.params[amplitude_key].value
                
                peak_data.append({
                    'Peak': peak_num,
                    'Center_cm-1': center,
                    'Center_Error': center_err,
                    'Sigma_cm-1': sigma,
                    'FWHM_cm-1': 2.355 * sigma,
                    'Amplitude': amplitude,
                })
        
        df_peaks = pd.DataFrame(peak_data)
        df_peaks.to_csv(f'{output_prefix}_peaks.csv', index=False)
        print(f"   ✅ {output_prefix}_peaks.csv")
        
        # Spectrum data
        df_spectrum = pd.DataFrame({
            'Raman_Shift_cm-1': self.x,
            'Intensity_Raw': self.y,
            'Baseline': self.baseline,
            'Intensity_Corrected': self.y_baseline_corrected,
            'Fit_Total': self.fit_result.best_fit,
        })
        df_spectrum.to_csv(f'{output_prefix}_spectrum.csv', index=False)
        print(f"   ✅ {output_prefix}_spectrum.csv")
        
        # Config
        config_text = json.dumps(self.config, indent=2, default=str)
        with open(f'{output_prefix}_config.txt', 'w') as f:
            f.write("RAMAN FITTING CONFIGURATION\n")
            f.write("="*70 + "\n\n")
            f.write(config_text)
        print(f"   ✅ {output_prefix}_config.txt")
        
        # Fit report
        with open(f'{output_prefix}_report.txt', 'w') as f:
            f.write(self.fit_result.fit_report())
        print(f"   ✅ {output_prefix}_report.txt")

    # ================================================================
    # MAIN WORKFLOW
    # ================================================================
    
    def run_interactive(self):
        """Run complete interactive workflow"""
        import json
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*68 + "║")
        print("║" + "PROFESSIONAL RAMAN SPECTROSCOPY PEAK FITTING".center(68) + "║")
        print("║" + "Research-Grade Analysis with Full User Control".center(68) + "║")
        print("║" + " "*68 + "║")
        print("╚" + "="*68 + "╝\n")
        
        # Step 1: Load
        if not self.load_data():
            return
        
        # Step 2: Preprocess
        self.preprocess_interactive()
        
        # Visualize preprocessed data
        plt.figure(figsize=(12, 5))
        plt.plot(self.x, self.y, 'k-', alpha=0.5, lw=1, label='Raw')
        plt.plot(self.x, self.y_baseline_corrected, 'r-', lw=1.5, 
                label='Baseline-corrected')
        plt.xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
        plt.ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.title('Data After Preprocessing', fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # Step 3: Specify peaks
        peaks = self.specify_peaks_interactive()
        
        # Step 4: Background
        self.choose_background_model()
        
        # Step 5: Peak models
        peak_models = self.choose_peak_models(len(peaks))
        
        # Step 6: Build model
        model, params = self.build_model(peaks, peak_models)
        
        # Step 7: Fitting config
        self.configure_fitting()
        
        # Step 8: Fit
        if not self.fit(model, params):
            return
        
        # Step 9: Results
        self.print_results()
        
        # Step 10: Plots
        self.plot_results()
        
        # Step 11: Export
        output_prefix = input("\nOutput file prefix [default='raman_fit']: ").strip() or 'raman_fit'
        self.export_results(output_prefix)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved with prefix: {output_prefix}")


if __name__ == "__main__":
    fitter = RamanPeakFitter()
    fitter.run_interactive()
