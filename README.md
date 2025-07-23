# 🏈 NFL Player Statistics Data Pipeline

This repository contains a full pipeline for gathering, cleaning, and preprocessing historical and current NFL player statistics for machine learning tasks.

Built with [`nfl_data_py`](https://pypi.org/project/nfl-data-py/), the pipeline supports seasonal and weekly data going back to **1999**, with automatic feature engineering and export of processed datasets in both Parquet and CSV formats.

---

## 📦 Features

- Fetches player statistics across **passing**, **rushing**, and **receiving** categories.
- Includes **roster metadata** (e.g., position, team, birth date).
- Supports both **seasonal** and **recent weekly** data.
- Performs:
  - Data cleaning
  - Feature engineering (e.g. age, per-game stats, years of experience)
  - One-hot encoding
  - Normalization of numerical features
- Exports both **interpretable** and **model-ready** datasets
- Logs all operations for transparency and debugging.

---

## 🧠 Use Cases

- Player performance modeling
- Fantasy football data science projects
- Career progression analysis
- Position-based clustering
- Sports betting model input

---

## ⚙️ Installation

```bash
git clone https://github.com/CarsonBoettinger/NFLPredictionModel.git
cd NFLPredictionModel
pip install -r requirements.txt
