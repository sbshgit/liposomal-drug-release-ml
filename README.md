# liposomal-drug-release-ml
Predictive Modeling of Drug Release from Liposomal Nano-Anticancer Drugs Using Advanced Machine Learning Techniques and TreeSHAP for Interpretability
# Predictive Modeling of Drug Release from Liposomal Nano-Anticancer Drugs

This repository contains all code and data supporting the article **“Predictive Modeling of Drug Release from Liposomal Nano-Anticancer Drugs Using Advanced Machine Learning Techniques and TreeSHAP for Interpretability”** by Saba Shiranirad _et al._ :contentReference[oaicite:0]{index=0}.

---

## 📖 Overview

In this work, we:

- Collected _in vitro_ release profiles for 7 unique liposomal anticancer formulations (480 measurements) by digitizing published plots.
- Calculated 24 input features, including physicochemical descriptors (via RDKit) and experimental conditions.
- Applied **nine** machine-learning algorithms (Lasso, PLS, SVR, k-NN, DT, RF, XGB, LGBM, NGB) within a nested cross-validation framework to predict fractional drug release.
- Identified **Natural Gradient Boosting (NGBRegressor)** as the best performer (MAE = 5.11).
- Used **TreeSHAP** to interpret feature contributions and highlight key drivers of release kinetics (e.g., % release at 1 h, time, drug-to-lipid ratio, concentration, encapsulation efficiency).

---

---

## 🛠️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/sbshgit/Predictive-Modeling-of-Drug-Release-from-Liposomal-Nano-Anticancer-Drugs.git
   cd Predictive-Modeling-of-Drug-Release-from-Liposomal-Nano-Anticancer-Drugs

## 📑 Citation
If you use this code or data, please cite:
S. Shiranirad and Z. Barzegar, "Predictive Modeling of Drug Release from Liposomal Nano-Anticancer Drugs Using Advanced Machine Learning Techniques and TreeSHAP for Interpretability," 2025 Eighth International Women in Data Science Conference at Prince Sultan University (WiDS PSU), Riyadh, Saudi Arabia, 2025, pp. 222-228, doi: 10.1109/WiDS-PSU64963.2025.00052. keywords: {Drugs;Adaptation models;Uncertainty;Accuracy;Machine learning;Predictive models;Mathematical models;Drug delivery;Data models;Kinetic theory;Liposomes;Artificial Intelligence;Machine learning;Drug release kinetics;SHAP analysis;Cancer therapy},

