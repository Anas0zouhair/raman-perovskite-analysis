"""
INTERACTIVE Raman Spectroscopy Analysis Tool
User-friendly interface that asks for all necessary parameters.

Run this script to get an interactive prompt asking for:
- File path
- Laser wavelength
- Input format (nm or cm-1)
- Preprocessing options
- Peak detection sensitivity
- Fitting model preferences
- Output preferences
"""

import numpy as np
import pandas as pd
from pathlib import Path
from universal_raman_advanced import universal_raman_analysis
import json

class RamanInteractiveAnalysis:
    """Interactive interface for Raman spectroscopy analysis"""
    
    def __init__(self):
        self.config = {}
        self.x = None
        self.y = None
        self.output_dir = Path("results")
        self.output_dir.mkdir(exist_ok=True)
    
    def print_header(self):
        """Print welcome message"""
        print("\n" + "="*80)
        print("╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "INTERACTIVE RAMAN SPECTROSCOPY ANALYSIS TOOL".center(78) + "║")
        print("║" + "Universal Pipeline for Halide Perovskites & All Materials".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝")
        print("="*80 + "\n")
    
    def step1_load_data(self):
        """Step 1: Load spectrum file"""
        print("\n" + "─"*80)
        print("STEP 1: LOAD YOUR SPECTRUM DATA")
        print("─"*80)
        print("\nSupported formats: .txt, .csv (tab or comma-delimited)")
        print("Expected format: 2 columns [wavelength/raman_shift, intensity]\n")
        
        while True:
            filepath = input("📁 Enter path to spectrum file: ").strip()
            filepath = Path(filepath)
            
            if not filepath.exists():
                print(f"❌ File not found: {filepath}")
                continue
            
            try:
                if filepath.suffix == '.csv':
                    delimiter = ','
                else:
                    delimiter = '\t'
                
                data = np.loadtxt(filepath, delimiter=delimiter)
                
                if data.shape[1] != 2:
                    print(f"❌ Expected 2 columns, got {data.shape[1]}")
                    continue
                
                self.x_raw = data[:, 0]
                self.y = data[:, 1]
                
                print(f"\n✅ Successfully loaded {filepath.name}")
                print(f"   Data points: {len(self.x_raw)}")
                print(f"   X range: {self.x_raw.min():.2f} - {self.x_raw.max():.2f}")
                print(f"   Y range: {self.y.min():.2f} - {self.y.max():.2f}")
                
                return filepath.stem  # Return filename for output prefix
                
            except Exception as e:
                print(f"❌ Error loading file: {e}")
    
    def step2_wavelength_format(self):
        """Step 2: Determine wavelength format"""
        print("\n" + "─"*80)
        print("STEP 2: WAVELENGTH FORMAT")
        print("─"*80)
        print("\nYour spectrum X-axis values range from {:.2f} to {:.2f}".format(
            self.x_raw.min(), self.x_raw.max()
        ))
        print("\nOptions:")
        print("  1) Nanometers (nm) - typically 300-1000 nm range")
        print("  2) Raman Shift (cm⁻¹) - typically 0-4000 cm⁻¹ range")
        print("  3) Not sure - auto-detect")
        
        while True:
            choice = input("\n🔍 Select format (1/2/3): ").strip()
            
            if choice == '1':
                self.x_format = 'nm'
                print("✅ Format: Nanometers (nm)")
                return True
            elif choice == '2':
                self.x_format = 'cm-1'
                print("✅ Format: Raman Shift (cm⁻¹)")
                self.x = self.x_raw.copy()
                return False  # No conversion needed
            elif choice == '3':
                # Auto-detect
                if self.x_raw.max() > 100:
                    self.x_format = 'nm'
                    print("✅ Auto-detected: Nanometers (nm)")
                    return True
                else:
                    self.x_format = 'cm-1'
                    print("✅ Auto-detected: Raman Shift (cm⁻¹)")
                    self.x = self.x_raw.copy()
                    return False
            else:
                print("❌ Invalid choice. Enter 1, 2, or 3")
    
    def step3_laser_wavelength(self):
        """Step 3: Get laser wavelength"""
        print("\n" + "─"*80)
        print("STEP 3: LASER WAVELENGTH (for nm → cm⁻¹ conversion)")
        print("─"*80)
        print("\nCommon laser wavelengths:")
        print("  - 405 nm (violet diode)")
        print("  - 532 nm (green Nd:YAG) ← MOST COMMON")
        print("  - 633 nm (red HeNe)")
        print("  - 785 nm (near-IR diode)")
        print("  - 1064 nm (IR Nd:YAG)")
        
        while True:
            wavelength_str = input("\n📍 Enter laser wavelength (nm): ").strip()
            try:
                self.laser_wavelength = float(wavelength_str)
                
                if self.laser_wavelength < 200 or self.laser_wavelength > 2000:
                    print("⚠️  Unusual wavelength. Are you sure?")
                    confirm = input("Continue? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                print(f"✅ Laser wavelength: {self.laser_wavelength} nm")
                return
            except:
                print("❌ Invalid input. Enter a number.")
    
    def convert_wavelength_to_raman(self):
        """Convert nm to Raman shift"""
        print("\n🔄 Converting wavelength to Raman shift...")
        
        # Sort by wavelength first
        sort_idx = np.argsort(self.x_raw)
        x_nm_sorted = self.x_raw[sort_idx]
        y_sorted = self.y[sort_idx]
        
        # Convert to Raman shift
        self.x = ((1.0 / self.laser_wavelength) - (1.0 / x_nm_sorted)) * 1e7
        self.y = y_sorted
        
        # Sort by Raman shift (ascending)
        sort_idx2 = np.argsort(self.x)
        self.x = self.x[sort_idx2]
        self.y = self.y[sort_idx2]
        
        print(f"✅ Raman shift range: {self.x.min():.1f} - {self.x.max():.1f} cm⁻¹")
    
    def step4_preprocessing_options(self):
        """Step 4: Preprocessing parameter selection"""
        print("\n" + "─"*80)
        print("STEP 4: PREPROCESSING OPTIONS")
        print("─"*80)
        
        print("\n🔹 COSMIC RAY REMOVAL")
        print("   Removes high-frequency spikes from cosmic rays")
        print("   Typical range: 3-7 (higher = less aggressive)")
        print("   Recommendation: 5 for most data")
        
        while True:
            cosmic_str = input("\n   Cosmic ray threshold (3-7) [default=5]: ").strip()
            if cosmic_str == "":
                self.cosmic_threshold = 5
                break
            try:
                self.cosmic_threshold = float(cosmic_str)
                if 1 <= self.cosmic_threshold <= 10:
                    break
                print("   ❌ Enter value between 3-7")
            except:
                print("   ❌ Invalid input")
        
        print(f"   ✅ Cosmic ray threshold: {self.cosmic_threshold}")
        
        print("\n🔹 BASELINE SUBTRACTION (airPLS)")
        print("   Controls smoothness of baseline removal")
        print("   λ = 1e4   : More baseline structure retained")
        print("   λ = 1e6   : Balanced (RECOMMENDED)")
        print("   λ = 1e8   : Very smooth baseline")
        print("   Typical range: 1e5 to 1e7")
        
        while True:
            baseline_str = input("\n   Baseline λ parameter [default=1e6]: ").strip()
            if baseline_str == "":
                self.airpls_lambda = 1e6
                break
            try:
                # Handle scientific notation
                self.airpls_lambda = float(baseline_str)
                if 1e3 <= self.airpls_lambda <= 1e9:
                    break
                print("   ❌ Enter value between 1e3 and 1e9")
            except:
                print("   ❌ Invalid input (e.g., 1e6, 1000000)")
        
        print(f"   ✅ Baseline λ: {self.airpls_lambda:.0e}")
        
        print("\n🔹 SUMMARY")
        print(f"   Cosmic ray threshold: {self.cosmic_threshold}")
        print(f"   Baseline λ: {self.airpls_lambda:.0e}")
    
    def step5_peak_detection_sensitivity(self):
        """Step 5: Peak detection sensitivity"""
        print("\n" + "─"*80)
        print("STEP 5: PEAK DETECTION SENSITIVITY")
        print("─"*80)
        print("\nThe algorithm automatically detects peaks.")
        print("Higher values = fewer, stronger peaks detected")
        print("Lower values = more peaks, including weak ones")
        
        print("\n  0.05  : Very sensitive (detects weak shoulders)")
        print("  0.10  : Sensitive (default, for most data)")
        print("  0.15  : Moderate (filters out weak peaks)")
        print("  0.20  : Conservative (only strong peaks)")
        
        while True:
            sens_str = input("\nPeak prominence ratio (0.05-0.25) [default=0.15]: ").strip()
            if sens_str == "":
                self.prominence_ratio = 0.15
                break
            try:
                self.prominence_ratio = float(sens_str)
                if 0.01 <= self.prominence_ratio <= 0.5:
                    break
                print("❌ Enter value between 0.05-0.25")
            except:
                print("❌ Invalid input")
        
        print(f"✅ Peak prominence ratio: {self.prominence_ratio}")
    
    def step6_model_selection(self):
        """Step 6: Peak model selection"""
        print("\n" + "─"*80)
        print("STEP 6: PEAK MODEL SELECTION")
        print("─"*80)
        print("\nChoose how the algorithm selects models for each peak:")
        
        print("\n  AUTO   : Intelligent selection (RECOMMENDED)")
        print("           Analyzes each peak shape and picks best model")
        print("           Models: Voigt, Lorentzian, PseudoVoigt, Gaussian")
        
        print("\n  VOIGT  : Best overall choice")
        print("           Convolution of Gaussian and Lorentzian")
        print("           Good for most halide perovskites")
        
        print("\n  LORENTZIAN : Sharp peaks")
        print("           Homogeneous broadening, narrow linewidths")
        print("           Good for well-defined crystal structures")
        
        print("\n  GAUSSIAN : Broad peaks")
        print("           Inhomogeneous broadening")
        print("           Good for disordered materials")
        
        while True:
            model_choice = input("\nSelect model type (AUTO/VOIGT/LORENTZIAN/GAUSSIAN) [default=AUTO]: ").strip().upper()
            
            if model_choice == "" or model_choice == "AUTO":
                self.model_type = "auto"
                print("✅ Model: AUTO (adaptive)")
                break
            elif model_choice in ["VOIGT", "LORENTZIAN", "GAUSSIAN"]:
                self.model_type = model_choice.lower()
                print(f"✅ Model: {model_choice}")
                break
            else:
                print("❌ Invalid choice")
    
    def step7_output_options(self):
        """Step 7: Output preferences"""
        print("\n" + "─"*80)
        print("STEP 7: OUTPUT OPTIONS")
        print("─"*80)
        
        while True:
            prefix = input("\nOutput file prefix [default='raman_analysis']: ").strip()
            if prefix == "":
                prefix = "raman_analysis"
            
            # Check for invalid characters
            if not all(c.isalnum() or c in "_- " for c in prefix):
                print("❌ Use only alphanumeric characters, spaces, dashes, underscores")
                continue
            
            self.output_prefix = str(self.output_dir / prefix)
            print(f"✅ Output prefix: {self.output_prefix}")
            break
        
        print("\n📊 Output files will include:")
        print("   - {prefix}_metrics.csv      (peak parameters)")
        print("   - {prefix}_spectrum.csv     (spectral data)")
        print("   - {prefix}_report.txt       (detailed fit report)")
        print("   - {prefix}.png              (plot, 300 DPI)")
        print("   - {prefix}.pdf              (plot, vector format)")
    
    def step8_review_parameters(self):
        """Step 8: Review all parameters"""
        print("\n" + "─"*80)
        print("STEP 8: REVIEW PARAMETERS")
        print("─"*80)
        
        print("\n📋 ANALYSIS CONFIGURATION:")
        print(f"   Wavelength format:        {self.x_format.upper()}")
        if self.x_format == 'nm':
            print(f"   Laser wavelength:         {self.laser_wavelength} nm")
            print(f"   Data range:               {self.x.min():.1f} - {self.x.max():.1f} cm⁻¹")
        print(f"   Data points:              {len(self.y)}")
        print(f"   Intensity range:          {self.y.min():.2f} - {self.y.max():.2f} a.u.")
        
        print(f"\n🔧 PREPROCESSING:")
        print(f"   Cosmic ray threshold:     {self.cosmic_threshold}")
        print(f"   Baseline λ:               {self.airpls_lambda:.0e}")
        
        print(f"\n🎯 PEAK DETECTION:")
        print(f"   Prominence ratio:         {self.prominence_ratio}")
        print(f"   Model selection:          {self.model_type.upper()}")
        
        print(f"\n💾 OUTPUT:")
        print(f"   File prefix:              {self.output_prefix}")
        
        while True:
            confirm = input("\n✅ Continue with analysis? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                print("\n⏭️  Please restart the script to modify parameters")
                return False
            else:
                print("❌ Enter 'y' or 'n'")
    
    def run_analysis(self):
        """Execute the analysis"""
        print("\n" + "─"*80)
        print("RUNNING ANALYSIS...")
        print("─"*80)
        
        try:
            result, metrics = universal_raman_analysis(
                self.x, self.y,
                output_prefix=self.output_prefix,
                plot=True
            )
            
            if result is not None:
                print("\n" + "─"*80)
                print("✅ ANALYSIS SUCCESSFUL!")
                print("─"*80)
                print(f"\n📁 Results saved to:")
                print(f"   {self.output_prefix}_metrics.csv")
                print(f"   {self.output_prefix}_spectrum.csv")
                print(f"   {self.output_prefix}_report.txt")
                print(f"   {self.output_prefix}.png")
                print(f"   {self.output_prefix}.pdf")
                
                print(f"\n📊 PEAK METRICS:")
                print(metrics.to_string(index=False))
                
                return result, metrics
            else:
                print("\n❌ Analysis failed. Check your data and parameters.")
                return None, None
                
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def run_interactive(self):
        """Run complete interactive workflow"""
        self.print_header()
        
        # Step 1: Load data
        filename = self.step1_load_data()
        
        # Step 2: Wavelength format
        needs_conversion = self.step2_wavelength_format()
        
        # Step 3: Laser wavelength (if needed)
        if needs_conversion:
            self.step3_laser_wavelength()
            self.convert_wavelength_to_raman()
        
        # Step 4: Preprocessing
        self.step4_preprocessing_options()
        
        # Step 5: Peak detection
        self.step5_peak_detection_sensitivity()
        
        # Step 6: Model selection
        self.step6_model_selection()
        
        # Step 7: Output
        self.step7_output_options()
        
        # Step 8: Review
        if not self.step8_review_parameters():
            return
        
        # Run analysis
        result, metrics = self.run_analysis()
        
        if result is not None:
            print("\n" + "="*80)
            print("🎉 THANK YOU FOR USING RAMAN ANALYSIS TOOL!")
            print("="*80 + "\n")

def main():
    """Main entry point"""
    analyzer = RamanInteractiveAnalysis()
    analyzer.run_interactive()

if __name__ == "__main__":
    main()
