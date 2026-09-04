"""
Example usage of the Raman spectroscopy analysis pipeline for halide perovskites.

This script demonstrates:
1. Creating configurations for different perovskite types
2. Running complete analysis pipeline
3. Customizing preprocessing parameters
4. Extracting and visualizing results
"""

from raman_analysis import (
    RamanConfig, 
    analyze_raman_spectrum,
    load_spectrum,
    preprocess_spectrum,
    build_model_from_config,
    fit_spectrum,
    extract_peak_metrics,
    export_results,
    plot_spectrum
)
import numpy as np

# ==========================================
# Example 1: Quick Analysis (Recommended)
# ==========================================

def example_quick_analysis():
    """
    Simplest way: use pre-made configuration and analyze in one line
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Analysis with Preset Config")
    print("="*60)
    
    # For mixed halides (Br/I):
    result, metrics = analyze_raman_spectrum(
        spectrum_file="sample_spectra/perovskite_spectrum.txt",
        config_file="config_perovskite_mixed_halides.json",
        output_prefix="results/mixed_halides"
    )
    
    print("\nPeak metrics:")
    print(metrics)
    
    return result, metrics

# ==========================================
# Example 2: Manual Configuration
# ==========================================

def example_manual_config():
    """
    Build configuration programmatically for custom analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Manual Configuration")
    print("="*60)
    
    # Create config
    config = RamanConfig()
    config.laser_wavelength_nm = 532.0
    config.airpls_lambda = 1e6
    config.cosmic_ray_threshold = 5
    
    # Add peaks manually
    config.add_peak(
        name="PbI_stretch",
        center=110,
        amplitude=1000,
        sigma=12,
        center_min=100,
        center_max=120,
        model_type="Lorentzian"
    )
    config.add_peak(
        name="PbBr_stretch",
        center=160,
        amplitude=1500,
        sigma=15,
        center_min=150,
        center_max=170,
        model_type="Lorentzian"
    )
    
    # Save for future use
    config.save_to_json("my_custom_config.json")
    print("\n[INFO] Config saved to 'my_custom_config.json'")
    
    # Run analysis
    result, metrics = analyze_raman_spectrum(
        spectrum_file="sample_spectra/perovskite_spectrum.txt",
        config_file="my_custom_config.json",
        output_prefix="results/custom_analysis"
    )
    
    return result, metrics

# ==========================================
# Example 3: Step-by-Step Control
# ==========================================

def example_step_by_step():
    """
    Manual pipeline for advanced users with full control
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Step-by-Step Pipeline")
    print("="*60)
    
    # Step 1: Load config
    config = RamanConfig("config_perovskite_mixed_cations.json")
    print(f"\n[Step 1] Config loaded: {len(config.peaks)} peaks")
    
    # Step 2: Load spectrum
    x, y_raw = load_spectrum(
        "sample_spectra/perovskite_spectrum.txt",
        config.laser_wavelength_nm
    )
    print(f"[Step 2] Spectrum loaded: {len(x)} points")
    print(f"         Range: {x.min():.1f} - {x.max():.1f} cm⁻¹")
    
    # Step 3: Preprocess
    y_smooth, baseline = preprocess_spectrum(y_raw, config)
    print(f"[Step 3] Preprocessing complete")
    
    # Step 4: Build model
    model, pars = build_model_from_config(config)
    print(f"[Step 4] Model built")
    
    # Step 5: Fit
    result = fit_spectrum(x, y_smooth, model, pars)
    print(f"[Step 5] Fitting complete (R² = {result.rsquared:.4f})")
    
    # Step 6: Extract metrics
    df_metrics = extract_peak_metrics(result, config)
    print(f"[Step 6] Metrics extracted:")
    print(df_metrics.to_string(index=False))
    
    # Step 7: Export
    export_results(x, y_raw, y_smooth, result, df_metrics, "results/step_by_step")
    print(f"[Step 7] Results exported")
    
    # Step 8: Plot
    plot_spectrum(x, y_smooth, result, config, "results/step_by_step_plot")
    print(f"[Step 8] Plot generated")
    
    return result, df_metrics

# ==========================================
# Example 4: Sensitivity Analysis
# ==========================================

def example_preprocessing_comparison():
    """
    Compare different preprocessing parameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Preprocessing Parameter Comparison")
    print("="*60)
    
    # Load spectrum
    x, y_raw = load_spectrum(
        "sample_spectra/perovskite_spectrum.txt",
        532.0
    )
    
    # Test different airpls_lambda values
    lambda_values = [1e4, 1e5, 1e6, 1e7]
    
    print("\nTesting baseline subtraction with different λ values:")
    for lam in lambda_values:
        config = RamanConfig()
        config.airpls_lambda = lam
        y_smooth, _ = preprocess_spectrum(y_raw, config)
        print(f"  λ = {lam:.0e}: min={y_smooth.min():.1f}, max={y_smooth.max():.1f}")
    
    # Use best value
    config = RamanConfig()
    config.airpls_lambda = 1e6
    y_smooth, _ = preprocess_spectrum(y_raw, config)
    
    return y_smooth

# ==========================================
# Example 5: Batch Processing Multiple Spectra
# ==========================================

def example_batch_processing():
    """
    Analyze multiple spectra with same configuration
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)
    
    import os
    from pathlib import Path
    
    config_file = "config_perovskite_mixed_halides.json"
    spectrum_dir = "sample_spectra/"
    results = {}
    
    # Process all .txt files in directory
    spectrum_files = sorted(Path(spectrum_dir).glob("*.txt"))
    print(f"\nFound {len(spectrum_files)} spectrum files")
    
    for i, spectrum_file in enumerate(spectrum_files, 1):
        output_prefix = f"results/batch_{spectrum_file.stem}"
        print(f"\n[{i}/{len(spectrum_files)}] Processing {spectrum_file.name}...")
        
        try:
            result, metrics = analyze_raman_spectrum(
                spectrum_file=str(spectrum_file),
                config_file=config_file,
                output_prefix=output_prefix
            )
            results[spectrum_file.name] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return results

# ==========================================
# Run Examples
# ==========================================

if __name__ == "__main__":
    
    # Uncomment to run examples (requires sample data)
    
    # example_quick_analysis()
    # example_manual_config()
    # example_step_by_step()
    # example_preprocessing_comparison()
    # example_batch_processing()
    
    print("""
    Available examples:
    1. example_quick_analysis()      - Fastest, uses preset config
    2. example_manual_config()       - Build config programmatically
    3. example_step_by_step()        - Full control over pipeline
    4. example_preprocessing_comparison() - Parameter sensitivity
    5. example_batch_processing()    - Analyze multiple spectra
    
    To run, uncomment in __main__ or import and call directly.
    """)
