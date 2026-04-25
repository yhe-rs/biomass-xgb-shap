# Forest AGB Estimation: Sentinel-2 & PALSAR-2 Time Series Integration

This repository contains the official Python implementation for the research paper:  
**"Forest Aboveground Biomass Estimation through Integration of Sentinel-2 and PALSAR-2 Time Series: Assessing Models Trained on GEDI and Field Inventory Benchmarks"** ([DOI: 10.1016/j.isprsjprs.2026.04.022](https://doi.org/10.1016/j.isprsjprs.2026.04.022))

The framework provides a comprehensive suite of tools to evaluate multi-source satellite data fusion and temporal feature engineering for high-accuracy biomass mapping across diverse forest ecosystems.

---

### 🔬 Key Research Objectives

#### 1. Temporal Feature & Sensor Configuration Analysis
We investigate how different data representations influence predictive accuracy:
* **Temporal Representations:** Comparison of static annual means, quarterly averages, and multi-temporal annual summaries to capture phenological variations.
* **Sensor Fusion:** Evaluation of **Sentinel-2** (optical/red-edge), **PALSAR-2** (L-band SAR), and their synergistic integration to mitigate saturation effects in high-biomass regions.

#### 2. Benchmark Source Evaluation
A critical assessment of training data reliability, comparing:
* **GEDI-Derived Products:** Utilizing spaceborne LiDAR (GEDI L4A) as a high-density, spatially continuous training source.
* **Field Inventory Benchmarks:** Validating model performance against ground-truth measurements to quantify bias and precision in satellite-based estimates.

#### 3. Model Robustness Across Environmental Gradients
To ensure global applicability, the framework assesses model stability across complex variables:
* **Topographic Complexity:** Analyzing performance variations across diverse **Elevation** ranges and **Slope** steepness.
* **Forest Structure:** Evaluating estimation error across a spectrum of **Forest Densities** and land-cover types to identify model sensitivity limits.

---

### 🚀 Technical Implementation
* **Modeling Architectures:** Implementation includes Gradient Boosting (**XGBoost**), Recurrent Neural Networks (**GRU/LSTM**), and **Attention-based** Transformer models.
* **Processing Framework:** Scalable data engineering utilizing **Google Earth Engine (GEE)** for imagery retrieval and **PyTorch** for deep learning training.
* **Error Propagation:** Includes a sequential Monte Carlo framework for rigorous uncertainty analysis.

---

### 📝 Citation
If you use this code or our findings in your research, please cite:
> *Wang et al. (2026). Forest Aboveground Biomass Estimation through Integration of Sentinel-2 and PALSAR-2 Time Series. ISPRS Journal of Photogrammetry and Remote Sensing.*
<img width="1024" height="1536" alt="agb" src="https://github.com/user-attachments/assets/8e3db1c4-edab-48dc-bf38-701ba511dcba" />
