# Movie Recommendation System

This project aims to improve content discovery and user engagement on movie platforms by building a robust, personalized recommender system. It integrates content-based filtering, collaborative filtering, and deep learning approaches to address cold-start, sparsity, and ranking challenges.

## Abstract

In the era of digital streaming and overwhelming content choices, our goal is to enhance user satisfaction by recommending movies that align with individual preferences. We tackled challenges such as item cold-start, data sparsity, and training inefficiencies using three approaches:

- **ScaNN-based Content Filtering** for fast semantic similarity search
- **LightFM Hybrid Filtering** for cold-start resilience
- **NeuMF (Neural Matrix Factorization)** to capture both linear and nonlinear dynamics

Evaluation was conducted using Top-K metrics and a realistic per-user temporal split.

## Datasets and Feature Engineering

We integrated two datasets:

- **TMDB 5000 Movie Metadata** (features: genre, cast, crew, company, language) [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata]
- **Kaggle Movies Dataset** (26M+ user interactions, ratings, and timestamps) [https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=links.csv]

Key engineered features include:
- **Genre**: One-hot encoded
- **Weighted Vote Score**: Combines average rating and vote count
- **Director/Cast/Company/Language Embeddings**: 300D semantic vectors via spaCy
- **Dimensionality Reduction**: PCA and TruncatedSVD applied to 1500D vectors

User filtering removed users with <5 ratings or zero variance to ensure data quality.

## Model 1: ScaNN Content Filtering

- **Inputs**: Weighted and normalized embeddings (genre, cast, director, company, language)
- **Embedding Weights**: Genre & cast (1.2), director (0.7), others (1.0)
- **Technique**: PCA (100D) + ScaNN with tree-structured search & asymmetric hashing

### Evaluation:
- Train Precision@10: 2.94%, NDCG@10: 0.0477
- Test Precision@10: 1.24%, NDCG@10: 0.0326

**Insight**: Efficient but low personalization. Lacks collaborative signals.

## Model 2: LightFM Hybrid Recommendation

- **Inputs**: Ratings ≥ 4 as implicit positives, SVD-compressed 446D item metadata
- **Loss**: WARP (Weighted Approximate-Rank Pairwise)
- **Optimizer**: Adagrad, 20 latent dimensions, 15 epochs

### Evaluation:
- Train Precision@10: 31.63%, NDCG@10: 0.5309
- Test Precision@10: 13.09%, NDCG@10: 0.2064, AUC: 87.81%

**Insight**: High hit rate and recall. Some overfitting observed. Good cold-start handling.

## Model 3: NeuMF (GMF + MLP + Metadata)

- **Architecture**:
  - GMF: Element-wise user-item embeddings
  - MLP: Concatenated embeddings + ReLU layers
  - Metadata fusion (genre, director, cast, etc.) via additive/concat pathways
- **Training**: PyTorch, MSE loss, Adam optimizer, batch size 512

### Evaluation:
- RMSE: 0.8813
- Train Precision@10: 9.83%, NDCG@10: 0.1125
- Test Precision@10: 10.48%, NDCG@10: 0.1885, AUC: 82.36%

**Insight**: Robust generalization and cold-start improvement via side features.

## System Setup

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib tensorflow keras torch scann lightfm tqdm surprise nltk
```

### Execution Order
1. `step_1_data_preprocess.ipynb`
2. `step_2_Content_ScaNN.ipynb`
3. `step_3_LightFM.ipynb`
4. `step_4_NeuMF.ipynb`

## 📊 Evaluation Metrics Used
- **Precision@K**, **Recall@K**
- **NDCG**, **MAP**, **MRR**, **Hit Rate@K**
- **RMSE** for rating prediction
- **AUC** for ranking evaluation

## 🧠 Contributions & Design Decisions
- Used **high-quality, popular Kaggle datasets** linked via TMDB IDs
- Applied **semantic feature engineering** and **filtering logic** to improve data quality
- Tuned model architectures for **real-world deployment feasibility** and **efficiency**
- Simulated production-like evaluation via **per-user temporal split**

## 📌 Future Enhancements
- [ ] Combine model predictions via weighted ensemble
- [ ] Deploy model outputs via Flask or Streamlit
- [ ] Add additional metadata (e.g., plot summaries, reviews)
- [ ] Integrate temporal user modeling
- [ ] Visualize embeddings via UMAP/t-SNE

## 🔗 References
1. He et al. (2017). Neural Collaborative Filtering
2. Kula (2015). Metadata Embeddings for Cold-Start
3. Google Research (2020). ScaNN
4. Weston et al. (2011). WARP loss
5. Duchi et al. (2011). Adagrad
6. Lops et al. (2011). Recommender Systems Handbook
7. IBM (2024). Content-Based Filtering Overview

