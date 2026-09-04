# Raman Spectroscopy Analysis for Halide Perovskites

A comprehensive, production-ready Python pipeline for Raman spectroscopy analysis of halide perovskites with mixed cations and halides.

## Features

✨ **Signal Preprocessing**
- Cosmic ray spike detection and removal using derivative analysis
- Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) baseline subtraction
- Savitzky-Golay smoothing for noise reduction

📊 **Data Processing**
- Automatic wavelength-to-Raman shift conversion (nm → cm⁻¹)
- Support for any laser wavelength (532, 633, 785, etc.)
- Flexible data loading from tab-delimited files

🎯 **Generalized Multi-Peak Fitting**
- Support for Voigt, Lorentzian, Gaussian, and Breit-Wigner models
- JSON-based peak configuration for easy reproducibility
- Physical constraints on peak positions and parameters
- Non-linear least-squares optimization via `lmfit`

📈 **Results Export**
- Peak metrics table (position, FWHM, amplitude, height)
- Fitted spectrum data (raw, processed, total fit, components)
- Detailed fitting report with statistical analysis
- Publication-quality plots (PNG 300 DPI + PDF)

## Installation

```bash
git clone https://github.com/Anas0zouhair/raman-perovskite-analysis.git
cd raman-perovskite-analysis
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Create a tab-delimited text file with two columns:
```
Wavelength (nm)    Intensity (a.u.)
780.5              1250
780.6              1255
...
```

### 2. Create a Configuration File

```json
{
  "laser_wavelength_nm": 532.0,
  "cosmic_ray_threshold": 5,
  "airpls_lambda": 1e6,
  "savgol_window": 11,
  "savgol_polyorder": 2,
  "peaks": [
    {
      "name": "peak1",
      "center": 150,
      "amplitude": 1000,
      "sigma": 15,
      "center_min": 140,
      "center_max": 160,
      "model_type": "Lorentzian"
    }
  ]
}
```

### 3. Run Analysis

```python
from raman_analysis import analyze_raman_spectrum

result, metrics = analyze_raman_spectrum(
    spectrum_file="my_spectrum.txt",
    config_file="my_config.json",
    output_prefix="my_analysis"
)
```

## Halide Perovskite Configurations

### Mixed Halides (Br/I)
Use `config_perovskite_mixed_halides.json` for analysis of perovskites with mixed halide composition:
- Low-frequency Pb-halide stretching modes (~110-160 cm⁻¹)
- Halide vibrations sensitive to composition

### Mixed Cations (MA/FA/Cs)
Use `config_perovskite_mixed_cations.json` for cation identification:
- MA (Methylammonium) ~965 cm⁻¹
- FA (Formamidinium) ~1008 cm⁻¹
- Cs (Cesium) ~1048 cm⁻¹

## API Reference

### `RamanConfig`
Configuration manager for the analysis pipeline.

**Methods:**
- `__init__(config_file=None)` - Initialize with optional JSON config
- `load_from_json(filename)` - Load settings from JSON
- `save_to_json(filename)` - Save settings to JSON
- `add_peak(name, center, amplitude, sigma, ...)` - Add peak manually

### `load_spectrum(filename, laser_wavelength_nm)`
Load spectrum data and convert to Raman shift.

**Returns:** `(x_raman_shift, y_intensity)`

### `preprocess_spectrum(y, config)`
Apply complete preprocessing pipeline.

**Returns:** `(y_processed, baseline)`

### `build_model_from_config(config)`
Construct composite lmfit model from configuration.

**Returns:** `(model, parameters)`

### `fit_spectrum(x, y, model, pars)`
Perform non-linear least-squares fitting.

**Returns:** `ModelResult`

### `extract_peak_metrics(result, config)`
Extract fitted peak parameters as pandas DataFrame.

**Returns:** `pd.DataFrame`

### `export_results(x, y_raw, y_smooth, result, df_metrics, output_prefix)`
Export all results to CSV and text files.

### `plot_spectrum(x, y_smooth, result, config, output_prefix)`
Generate publication-quality figures.

### `analyze_raman_spectrum(spectrum_file, config_file, output_prefix)`
Complete end-to-end pipeline in one call.

## Output Files

For `output_prefix="my_analysis"`, generates:

- `my_analysis_peak_metrics.csv` - Peak parameters table
- `my_analysis_spectrum.csv` - Spectral data (raw, processed, fit)
- `my_analysis_fit_report.txt` - Detailed fit statistics
- `my_analysis.png` - High-resolution figure (300 DPI)
- `my_analysis.pdf` - Vector figure for publications

## Advanced Usage

### Custom Model Types

```python
config = RamanConfig()
config.add_peak(
    name="custom_peak",
    center=500,
    amplitude=2000,
    sigma=20,
    center_min=480,
    center_max=520,
    model_type="Gaussian"  # Choose: Voigt, Lorentzian, Gaussian, BreitWigner
)
```

### Adjusting Preprocessing

```python
config.cosmic_ray_threshold = 7  # Higher = less aggressive
config.airpls_lambda = 1e5       # Lower = less smooth baseline
config.savgol_window = 15        # Higher = more smoothing
```

## Theory & References

### Raman Shift Conversion
```
Δν (cm⁻¹) = (1/λ_laser - 1/λ_observed) × 10⁷
```

### airPLS Algorithm
Iteratively weighted least squares with sparsity-promoting penalty. Particularly useful for fluorescence background in perovskite Raman spectra.

**Reference:** Zhang et al., Analyst 2010, 135, 1138-1146

### Peak Models
- **Voigt**: Convolution of Gaussian and Lorentzian (good general choice)
- **Lorentzian**: Homogeneous broadening, sharp peaks
- **Gaussian**: Inhomogeneous broadening, broad peaks
- **Breit-Wigner**: Resonance line shapes

## Troubleshooting

**Issue:** Baseline oscillates after subtraction
- **Solution:** Increase `airpls_lambda` (1e7 or higher)

**Issue:** Peaks not fit well
- **Solution:** Adjust initial `center`, `amplitude`, `sigma` values
- **Solution:** Change `model_type` to better match peak shape
- **Solution:** Add `center_min`/`center_max` constraints

**Issue:** Cosmic rays not removed
- **Solution:** Increase `cosmic_ray_threshold` (5→7)

## Contributing

Contributions welcome! Areas for enhancement:
- Support for 2D peak mapping
- Temperature-dependent analysis automation
- Comparison tools for multiple spectra
- Advanced baseline correction methods

## License

MIT License - See LICENSE file for details

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{raman_perovskite_2024,
  author = {Anas Zouhair},
  title = {Raman Spectroscopy Analysis Pipeline for Halide Perovskites},
  year = {2024},
  url = {https://github.com/Anas0zouhair/raman-perovskite-analysis}
}
```

## Questions?

Open an issue on GitHub or contact the maintainer.

---

**Last Updated:** September 2024
**Version:** 1.0
