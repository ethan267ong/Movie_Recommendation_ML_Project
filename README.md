---
title: "Movie Recommendation System"
---

# 🎬 Movie Recommendation System

This project builds a robust and modular hybrid movie recommender system using content-based filtering, collaborative filtering, and neural collaborative filtering. It integrates both rich metadata and user behavior to generate personalized movie suggestions.

## 📁 Project Structure

### 1. `step_1_data_preprocess.ipynb` – Data Preprocessing
- Merges and cleans `ratings.csv`, `movies.csv`, and `tags.csv`, and maps them with the TMDB dataset.
- Engineers features from movie metadata: `director_embedding`, `main_cast_embeddings`, `production_company`, `original_language_embedding`.
- Filters out low-interaction users and low-information movies.
- Prepares train-test splits (including per-user holdouts).

> **Note:** Dataset integration links TMDB and The Movies Dataset via TMDB and IMDb IDs. Embeddings are normalized. A custom `weighted_vote_score` metric reflects item popularity.

---

### 2. `step_2_Content_ScaNN.ipynb` – Content-Based Filtering with ScaNN
- Constructs 1500-dimensional semantic embeddings using genre, cast, director, production company, and language vectors.
- Applies embedding weights: genre & cast (1.2), director (0.7), others (1.0).
- Reduces to 100D using PCA (~90% variance retained).
- Uses Google ScaNN for efficient approximate nearest-neighbor search.

> **Highlight:** Post-processing step ranks retrieved items by `weighted_vote_score` to prioritize quality.

**Evaluation Metrics:**

| Split | Precision@10 | NDCG@10 |
|-------|---------------|---------|
| Train | 2.94%         | 0.0477  |
| Test  | 1.24%         | 0.0326  |

> **Insight:** Pure content-based filtering suffers from limited personalization. Hybridization is needed.

---

### 3. `step_3_LightFM.ipynb` – Hybrid Collaborative Filtering with LightFM
- Builds user–movie interaction matrix (ratings ≥ 4 mapped to 1).
- Applies TruncatedSVD on content embeddings to reduce from 1500D to 446D.
- Trains using WARP loss to optimize Top-K relevance.
- Adagrad optimizer, 20 latent factors, 15 epochs.

> **Highlight:** Integrates metadata as side features, improving cold-start performance.

**Evaluation Metrics:**

| Split | Precision@10 | NDCG@10 |
|-------|---------------|---------|
| Train | 31.63%        | 0.5309  |
| Test  | 13.09%        | 0.2064  |

> **Insight:** Performs well overall. Signs of overfitting—further generalization methods recommended.

---

### 4. `step_4_NeuMF.ipynb` – Neural Collaborative Filtering (NeuMF)
- Combines Generalized Matrix Factorization (GMF) and MLP for deep preference modeling.
- Incorporates side metadata embeddings (genres, cast, directors, etc.) into the MLP path.
- Uses PyTorch, Adam optimizer, and MSE loss.

> **Highlight:** Uses three pathways: GMF, MLP, and metadata, enhancing representation and generalization.

**Evaluation Metrics:**

| Split | Precision@10 | NDCG@10 | AUC     |
|-------|---------------|---------|---------|
| Train | 9.83%         | 0.1125  | 75.82%  |
| Test  | 10.48%        | 0.1885  | 82.36%  |

> **Insight:** High recall and generalization. Cold-start robustness improved. Ranking could be further optimized.

---

## ⚙️ Setup

### Requirements

- Python 3.8+
- Jupyter Notebook
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- `tensorflow`, `keras`, `torch`
- `scann`, `lightfm`
- `tqdm`, `surprise`, `nltk`

### Installation

```bash
pip install -r requirements.txt
