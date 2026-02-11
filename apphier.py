# ============================================
# 🟣 News Topic Discovery Dashboard
# Hierarchical Clustering + TF-IDF + PCA
# ============================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group similar news articles "
    "based on textual similarity."
)

st.markdown("---")


# ============================================
# SIDEBAR CONTROLS
# ============================================

st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV dataset to begin.")
    st.stop()

# Load dataset safely
try:
    df = pd.read_csv(uploaded_file, encoding="latin1")
except:
    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

st.success("Dataset Loaded Successfully!")

st.write("### Dataset Preview")
st.dataframe(df.head())


# ============================================
# AUTO TEXT COLUMN DETECTION
# ============================================

st.sidebar.markdown("---")
st.sidebar.header("📝 Text Column Selection")

possible_text_cols = [col for col in df.columns if df[col].dtype == "object"]

if len(possible_text_cols) == 0:
    st.error("No text column found in dataset.")
    st.stop()

text_col = st.sidebar.selectbox("Select News Text Column", possible_text_cols)

texts = df[text_col].dropna().astype(str)

st.write(f"✅ Using Text Column: **{text_col}**")
st.write(f"Total Articles: {len(texts)}")


# ============================================
# TF-IDF SETTINGS
# ============================================

st.sidebar.markdown("---")
st.sidebar.header("🔠 TF-IDF Vectorization Controls")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)

stopwords = st.sidebar.checkbox("Remove English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)


# ============================================
# HIERARCHICAL CLUSTER SETTINGS
# ============================================

st.sidebar.markdown("---")
st.sidebar.header("🌳 Hierarchical Clustering Controls")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendro_articles = st.sidebar.slider(
    "Articles for Dendrogram (subset)",
    20, 200, 50
)

cluster_count = st.sidebar.slider(
    "Number of Clusters (after dendrogram)",
    2, 10, 5
)

cut_height = st.sidebar.slider(
    "Horizontal Cut Line Height",
    0, 500, 150
)


# ============================================
# STEP 1: TF-IDF VECTOR CREATION
# ============================================

st.markdown("## 1️⃣ Convert News Text into TF-IDF Features")

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if stopwords else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(texts)

st.success(f"TF-IDF Matrix Created: {X.shape[0]} articles × {X.shape[1]} features")


# ============================================
# STEP 2: DENDROGRAM BUTTON
# ============================================

st.markdown("---")
st.markdown("## 2️⃣ Dendrogram Construction")

if st.button("🟦 Generate Dendrogram"):

    subset = X[:dendro_articles].toarray()

    fig, ax = plt.subplots(figsize=(10, 5))

    dendrogram = sch.dendrogram(
        sch.linkage(subset, method=linkage_method)
    )

    # Horizontal cut line
    ax.axhline(y=cut_height, linestyle="--")

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Articles")
    plt.ylabel("Distance")

    st.pyplot(fig)

    st.info(
        "Large vertical gaps suggest strong topic separation. "
        "Choose cluster count based on the cut line."
    )


# ============================================
# STEP 3: APPLY CLUSTERING
# ============================================

st.markdown("---")
st.markdown("## 3️⃣ Apply Hierarchical Clustering")

if st.button("🟩 Apply Clustering"):

    hc = AgglomerativeClustering(
        n_clusters=cluster_count,
        metric="euclidean",
        linkage=linkage_method
    )

    labels = hc.fit_predict(X.toarray())

    df["Cluster"] = labels

    st.success("Clustering Completed Successfully!")

    # ============================================
    # STEP 4: SILHOUETTE SCORE
    # ============================================

    st.markdown("## 📊 Validation: Silhouette Score")

    if cluster_count > 1:
        score = silhouette_score(X, labels)
        st.write(f"✅ Silhouette Score: **{score:.3f}**")

        if score > 0.5:
            st.success("Clusters are well separated!")
        elif score > 0.2:
            st.warning("Clusters overlap slightly.")
        else:
            st.error("Poor clustering. Try changing linkage or cluster count.")

    # ============================================
    # STEP 5: PCA VISUALIZATION
    # ============================================

    st.markdown("---")
    st.markdown("## 4️⃣ PCA Cluster Visualization (2D)")

    X_dense = X.toarray()

    # Safe PCA
    max_possible = min(X_dense.shape[0], X_dense.shape[1])
    n_components = min(50, max_possible - 1)

    if n_components < 2:
        st.error("Dataset too small for PCA visualization.")
        st.stop()

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_dense)

    pca_df = pd.DataFrame()
    pca_df["PC1"] = X_2d[:, 0]
    pca_df["PC2"] = X_2d[:, 1]
    pca_df["Cluster"] = labels

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    scatter = ax2.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"]
    )

    plt.title("2D PCA Projection of News Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    st.pyplot(fig2)

    # ============================================
    # STEP 6: CLUSTER SUMMARY
    # ============================================

    st.markdown("---")
    st.markdown("## 5️⃣ Cluster Summary (Top Keywords)")

    terms = vectorizer.get_feature_names_out()

    summary = []

    for c in range(cluster_count):

        cluster_articles = df[df["Cluster"] == c][text_col]

        # Top keywords
        cluster_tfidf = X[df["Cluster"] == c].mean(axis=0)
        cluster_tfidf = np.asarray(cluster_tfidf).flatten()

        top_idx = cluster_tfidf.argsort()[-10:][::-1]
        top_words = [terms[i] for i in top_idx]

        snippet = cluster_articles.iloc[0][:150] if len(cluster_articles) > 0 else ""

        summary.append({
            "Cluster ID": c,
            "Articles": len(cluster_articles),
            "Top Keywords": ", ".join(top_words),
            "Sample Article": snippet
        })

    summary_df = pd.DataFrame(summary)

    st.dataframe(summary_df)

    # ============================================
    # STEP 7: BUSINESS INTERPRETATION
    # ============================================

    st.markdown("---")
    st.markdown("## 6️⃣ Business Interpretation")

    for row in summary:
        st.write(
            f"🟣 Cluster {row['Cluster ID']}: Articles mainly talk about "
            f"**{row['Top Keywords'].split(',')[0]}**, "
            "showing a common theme."
        )

    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, "
        "and better content organization."
    )
