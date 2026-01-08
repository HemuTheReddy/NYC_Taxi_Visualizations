# 🚕 Manhattan Minutes: Decoding the NYC Taxi Pulse

[![GCP Deployed](https://img.shields.io/badge/GCP-Cloud%20Run-blue?logo=google-cloud)](https://dashapp-965780251612.us-east1.run.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://www.python.org/)

This project was completed for the Fall 2025 Semester Term Project of CS5764. It is a comprehensive geospatial and statistical analysis of NYC Taxi trip data. It transforms raw trip records into a multi-layered interactive experience, combining backend statistical modeling with a production-ready data pipeline dashboard.

---

Dashboard GCP Link: (to be added)

---

## 🏗️ Project Architecture

The project consists of three core components:

1. **The Statistical Engine (`analysis.py`)**: A backend script dedicated to heavy data cleaning, feature engineering, and mathematical validation (PCA, Hypothesis Testing).
2. **The Visualization Gallery (`vis.py`)**: A thorough Exploratory Data Analysis (EDA) suite containing 20+ advanced plots ranging from 3D spatial clusters to dense hexbin distributions.
3. **The Interactive Dashboard (`app.py`)**: A Dash-based web application that allows users to iteratively clean, transform, and visualize the data in real-time.
4. **The Executive Report**: A formal synthesis of findings, methodology, and visualizations derived from the analysis.

---

## 🖥️ Dashboard Features

The dashboard functions as a modular pipeline, allowing for "What-If" scenarios:  
* **Real-time Cleaning:** Toggle checklists for dropping duplicates, nulls, and negative durations.
* **Dynamic Transform Gallery:** Observe the impact of Log, Sqrt, and Box-Cox transformations on distribution curves instantly.
* **PCA Explorer:** Adjust components via slider to visualize cumulative explained variance.
* **Multivariate Visualizations:** Explore 3D scatter plots, spatial KDE maps, and bivariate regression models.

---

## 🚀 Quick Start

1. Clone the repository:
```sh
!git clone https://github.com/HemuTheReddy/NYC_Taxi_Visualizations
```
2. Install required libraries
```bash
pip install dash pandas numpy plotly scipy sklearn seaborn
```
3. Run the Analysis
```bash
python analysis.py
```
4. Run the visualization script
```bash
python vis.py
```
5. Launch the dashboard
```bash
python app.py
```
