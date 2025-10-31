## BEED Training Needs Clustering Web App

An interactive Streamlit app to explore K-Means clustering of training needs for the BEED Extension Program. Upload an Excel dataset, compute clusters, inspect demographics and dominant needs per cluster, visualize PCA, and export labeled results.

### Quick Start

1) Install dependencies (prefer a virtual environment):

```bash
pip install -r requirements.txt
```

2) Place your dataset as `1dataset.xlsx` in the project folder or upload it from the app UI.

3) Run the app:

```bash
streamlit run app.py
```

4) Open the provided local URL in your browser.

### Dataset Expectations

- Training need columns are detected by the regex pattern `^\d+\.\d+\.\s` (e.g., `1.1. Classroom Management`). They should be numeric.
- Optional demographic columns: Age, Gender, Years of Experience. You can select these from the sidebar.

### Outputs

- Dashboard with: record count, feature count, silhouette scores across k, cluster distribution, PCA scatter, and average training needs per cluster.
- Cluster profiles: demographic summaries and dominant training needs per cluster.
- Export: `clustering_results.xlsx` with labeled data and cluster means.



