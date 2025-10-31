import io
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
	page_title="BEED Training Needs Analysis",
	layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes) -> pd.DataFrame:
	return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_default_data(default_path: str) -> Optional[pd.DataFrame]:
	if not os.path.exists(default_path):
		return None
	return pd.read_excel(default_path)


def generate_demo_data(num_rows: int = 150) -> pd.DataFrame:
	# Synthetic demographics
	rng = np.random.default_rng(42)
	age = rng.integers(22, 60, size=num_rows)
	years_exp = np.clip(age - rng.integers(18, 30, size=num_rows), 0, None)
	gender = rng.choice(["Female", "Male"], size=num_rows, p=[0.7, 0.3])

	# Synthetic training-need features (1-5 scale), lightly correlated with experience/age
	def bounded(v):
		return np.clip(v, 1, 5)

	base = rng.normal(3.2, 0.6, size=(num_rows,))
	cm = bounded(base + rng.normal(0, 0.5, num_rows) - 0.01 * (years_exp - years_exp.mean()))  # 1.1.
	pl = bounded(base + rng.normal(0, 0.5, num_rows) - 0.005 * (age - age.mean()))             # 1.2.
	di = bounded(base + rng.normal(0, 0.5, num_rows) + 0.008 * (years_exp.mean() - years_exp)) # 1.3.
	ict = bounded(base + rng.normal(0, 0.5, num_rows))                                         # 1.4.
	assess = bounded(base + rng.normal(0, 0.5, num_rows) - 0.004 * (age - age.mean()))         # 1.5.

	df_demo = pd.DataFrame(
		{
			"Age": age,
			"Gender": gender,
			"Years of Experience": years_exp,
			"1.1. Classroom Management": cm.round(2),
			"1.2. Pedagogical Literacy": pl.round(2),
			"1.3. Differentiated Instruction": di.round(2),
			"1.4. ICT Integration": ict.round(2),
			"1.5. Assessment Strategies": assess.round(2),
		}
	)
	return df_demo


def detect_training_need_columns(df: pd.DataFrame) -> List[str]:
	pattern = re.compile(r"^\d+\.\d+\.\s")
	return [c for c in df.columns if isinstance(c, str) and pattern.match(c)]


def safe_numeric_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
	numeric_cols = []
	for c in columns:
		if pd.api.types.is_numeric_dtype(df[c]):
			numeric_cols.append(c)
	return numeric_cols


def pick_default_demographic_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
	# Try common names; fall back to None
	columns_lower = {c.lower(): c for c in df.columns}
	age = None
	gender = None
	exp = None
	for key in ["age", "ages"]:
		if key in columns_lower:
			age = columns_lower[key]
			break
	for key in ["gender", "sex"]:
		if key in columns_lower:
			gender = columns_lower[key]
			break
	for key in ["years of experience", "experience", "years teaching", "years of teaching experience"]:
		if key in columns_lower:
			exp = columns_lower[key]
			break
	return age, gender, exp


def compute_optimal_k(x: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, List[float]]:
	scores: List[float] = []
	best_k = k_min
	best_score = -1.0
	for k in range(k_min, k_max + 1):
		kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
		labels = kmeans.fit_predict(x)
		score = silhouette_score(x, labels)
		scores.append(score)
		if score > best_score:
			best_score = score
			best_k = k
	return best_k, scores


def recommend_programs(cluster_means: pd.DataFrame, top_n: int = 5) -> Dict[int, List[str]]:
	# Recommend top-N highest mean training-need items per cluster
	recs: Dict[int, List[str]] = {}
	for cluster_id, row in cluster_means.iterrows():
		# Sort descending by need level
		top_items = row.sort_values(ascending=False).head(top_n).index.tolist()
		recs[int(cluster_id)] = top_items
	return recs


def _relative_descriptor(value: float, reference: float, tol: float = 0.1) -> str:
	if value >= reference * (1 + tol):
		return "higher than average"
	if value <= reference * (1 - tol):
		return "lower than average"
	return "about average"


def build_cluster_narrative(
	cluster_id: int,
	cluster_df: pd.DataFrame,
	overall: pd.DataFrame,
	cluster_means: pd.DataFrame,
	need_cols: List[str],
	age_col: Optional[str],
	gender_col: Optional[str],
	exp_col: Optional[str],
	top_n: int = 5,
) -> str:
	parts: List[str] = []

	# Demographics
	if age_col and age_col in cluster_df.columns and pd.api.types.is_numeric_dtype(cluster_df[age_col]):
		age_mean_c = float(cluster_df[age_col].mean())
		age_mean_all = float(overall[age_col].mean()) if age_col in overall.columns else age_mean_c
		parts.append(f"The average age is {age_mean_c:.1f} years, which is {_relative_descriptor(age_mean_c, age_mean_all)}.")

	if exp_col and exp_col in cluster_df.columns and pd.api.types.is_numeric_dtype(cluster_df[exp_col]):
		exp_mean_c = float(cluster_df[exp_col].mean())
		exp_mean_all = float(overall[exp_col].mean()) if exp_col in overall.columns else exp_mean_c
		parts.append(f"Average years of teaching experience is {exp_mean_c:.1f}, {_relative_descriptor(exp_mean_c, exp_mean_all)} vs. all teachers.")

	if gender_col and gender_col in cluster_df.columns:
		gender_counts = cluster_df[gender_col].value_counts(normalize=True).sort_values(ascending=False)
		if not gender_counts.empty:
			top_gender, pct = gender_counts.index[0], float(gender_counts.iloc[0] * 100)
			parts.append(f"Gender composition is led by {top_gender} (~{pct:.0f}%).")

	# Dominant needs
	if cluster_id in cluster_means.index:
		row = cluster_means.loc[cluster_id, need_cols]
		top_items = row.sort_values(ascending=False).head(top_n)
		need_list = ", ".join([f"{name} ({val:.2f})" for name, val in top_items.items()])
		parts.append(f"Dominant training needs: {need_list}.")
		parts.append("Recommendation: prioritize targeted activities addressing these top-rated needs for this cluster.")

	return " " .join(parts)


def render_sidebar(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str], Optional[str], int, int]:
	st.sidebar.header("Controls")
	st.sidebar.write("The dashboard uses '1dataset.xlsx' if present; otherwise demo data.")

	# Demographics selection
	age_default, gender_default, exp_default = pick_default_demographic_columns(df)
	col_options = [None] + list(df.columns)
	age_col = st.sidebar.selectbox("Age column (optional)", col_options, index=(col_options.index(age_default) if age_default in col_options else 0))
	gender_col = st.sidebar.selectbox("Gender column (optional)", col_options, index=(col_options.index(gender_default) if gender_default in col_options else 0))
	exp_col = st.sidebar.selectbox("Years of experience column (optional)", col_options, index=(col_options.index(exp_default) if exp_default in col_options else 0))

	# K range
	k_min = st.sidebar.number_input("Min clusters (k)", min_value=2, max_value=2, value=2, step=1, help="Silhouette search lower bound")
	k_max = st.sidebar.number_input("Max clusters (k)", min_value=3, max_value=15, value=10, step=1, help="Silhouette search upper bound")

	return "sidebar", age_col, gender_col, exp_col, int(k_min), int(k_max)


def main() -> None:
	st.title("BEED Training Needs Dashboard")
	st.caption("Development of a Web-Based Clustering Analysis of Training Needs for the BEED Extension Program")

	# Data source (fixed: no upload; use default file or demo)
	st.subheader("RECOMMENDED K VALUE IS 5")
	default_path = os.path.join(os.getcwd(), "1dataset.xlsx")
	data: Optional[pd.DataFrame] = load_default_data(default_path)
	if data is None:
		st.info("'1dataset.xlsx' not found. Using demo data so you can explore the app.")
		data = generate_demo_data(180)

	# Feature detection
	training_need_columns = detect_training_need_columns(data)
	if not training_need_columns:
		st.error("No training-need columns matched the pattern '^#.#. ' (e.g., '1.1. ...'). Please verify your column names.")
		st.dataframe(data.head())
		return

	# Keep only numeric TN columns
	numeric_tn_columns = safe_numeric_columns(data, training_need_columns)
	if not numeric_tn_columns:
		st.error("Detected training-need columns are not numeric. Please ensure they contain numeric scores.")
		return

	# Sidebar controls
	_, age_col, gender_col, exp_col, k_min, k_max = render_sidebar(data)

	# Preprocess
	features_df = data[numeric_tn_columns].copy()
	missing = features_df.isnull().sum()
	missing_total = int(missing.sum())
	if missing_total > 0:
		features_df = features_df.fillna(features_df.median(numeric_only=True))

	scaler = StandardScaler()
	scaled = scaler.fit_transform(features_df)

	# Determine optimal k via silhouette
	best_k, silhouette_scores = compute_optimal_k(scaled, k_min=k_min, k_max=k_max)

	# Allow override
	chosen_k = st.slider("Select number of clusters (k)", min_value=k_min, max_value=k_max, value=best_k)

	# KMeans fit
	kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
	labels = kmeans.fit_predict(scaled)

	data_clust = data.copy()
	data_clust["cluster_label"] = labels
	features_with_labels = pd.DataFrame(scaled, columns=numeric_tn_columns)
	features_with_labels["cluster_label"] = labels

	# Dashboard
	st.subheader("Dashboard")
	left, right = st.columns([1, 2])
	with left:
		st.metric("Records", len(data_clust))
		st.metric("Training-Need Variables", len(numeric_tn_columns))
		st.metric("Missing values imputed", missing_total)
		st.metric("Selected k", chosen_k)
	with right:
		st.write("Silhouette scores across k")
		ss_df = pd.DataFrame({"k": list(range(k_min, k_max + 1)), "score": silhouette_scores})
		st.line_chart(ss_df.set_index("k"))

	# Cluster distribution
	st.write("### Cluster Distribution")
	dist = data_clust["cluster_label"].value_counts().sort_index()
	dist_df = pd.DataFrame({"cluster": dist.index.astype(str), "count": dist.values})
	st.bar_chart(dist_df.set_index("cluster"))

	# Average training needs per cluster
	st.write("### Average Training Needs per Cluster (Top 10)")
	cluster_means_all = data_clust.groupby("cluster_label")[numeric_tn_columns].mean()
	# Show top-10 by overall variance/importance
	variances = features_df.var().sort_values(ascending=False)
	top_vars = list(variances.head(10).index)
	st.dataframe(cluster_means_all[top_vars].round(2))

	# PCA visualization
	st.write("### PCA Scatter Plot (2D)")
	pca = PCA(n_components=2, random_state=42)
	pca_points = pca.fit_transform(scaled)
	pca_df = pd.DataFrame(pca_points, columns=["PC1", "PC2"])
	pca_df["cluster_label"] = labels
	for c in sorted(pca_df["cluster_label"].unique()):
		subset = pca_df[pca_df["cluster_label"] == c]
		st.scatter_chart(subset, x="PC1", y="PC2")

	# Cluster Profiles
	st.subheader("Cluster Profiles")
	with st.expander("Demographic summaries", expanded=True):
		cols = st.columns(chosen_k)
		for idx, c in enumerate(sorted(data_clust["cluster_label"].unique())):
			with cols[idx % len(cols)]:
				st.write(f"Cluster {c}")
				cluster_df = data_clust[data_clust["cluster_label"] == c]
				if age_col and age_col in cluster_df.columns and pd.api.types.is_numeric_dtype(cluster_df[age_col]):
					st.write({"mean_age": float(cluster_df[age_col].mean()), "median_age": float(cluster_df[age_col].median())})
				if exp_col and exp_col in cluster_df.columns and pd.api.types.is_numeric_dtype(cluster_df[exp_col]):
					st.write({"mean_experience": float(cluster_df[exp_col].mean()), "median_experience": float(cluster_df[exp_col].median())})
				if gender_col and gender_col in cluster_df.columns:
					st.write("Gender distribution")
					st.dataframe(cluster_df[gender_col].value_counts().to_frame("count"))

	# Dominant needs per cluster
	st.write("### Dominant Training Needs per Cluster")
	cluster_means = cluster_means_all.copy()
	recs = recommend_programs(cluster_means, top_n=5)
	for c in sorted(recs.keys()):
		st.write(f"Cluster {c}: Top needs")
		st.write(recs[c])

	# Interpretations
	st.subheader("Cluster Interpretations")
	for c in sorted(data_clust["cluster_label"].unique()):
		cluster_df = data_clust[data_clust["cluster_label"] == c]
		narr = build_cluster_narrative(
			cluster_id=int(c),
			cluster_df=cluster_df,
			overall=data_clust,
			cluster_means=cluster_means,
			need_cols=numeric_tn_columns,
			age_col=age_col,
			gender_col=gender_col,
			exp_col=exp_col,
			top_n=5,
		)
		st.markdown(f"**Cluster {c} interpretation:** {narr}")

	# Downloadable results
	st.subheader("Export Results")
	with pd.ExcelWriter("clustering_results.xlsx", engine="openpyxl") as writer:
		data_clust.to_excel(writer, index=False, sheet_name="labeled_data")
		cluster_means_all.to_excel(writer, sheet_name="cluster_means")
		pd.DataFrame({"feature": numeric_tn_columns}).to_excel(writer, index=False, sheet_name="features")
	with open("clustering_results.xlsx", "rb") as f:
		st.download_button("Download labeled dataset & summaries", f, file_name="clustering_results.xlsx")

	st.info("Tip: Use the sidebar to set demographic columns and cluster search range.")


if __name__ == "__main__":
	main()


