# Autonomous Industrial Vehicle: ML Product Grouping

Final phase of a machine-learning project for an autonomous vehicle operating in an industrial plant. This phase shifts from supervised protocol prediction to unsupervised grouping of products for four destination depots.

## Project highlights

- Preprocessed product features with a scikit-learn `ColumnTransformer` pipeline, including scaling and one-hot encoding.
- Applied K-Means (`k=4`) to suggest depot assignments from product attributes.
- Calculated the silhouette score to assess the resulting clusters. The notebook does not record a final numeric silhouette result.

## Core competencies

Unsupervised learning, K-Means clustering, tabular data preprocessing, categorical encoding, feature scaling, clustering evaluation.

## Code

`src/cluster_products.py` is an export of the Phase III notebook's Python cells. Update the input data path to point to your own authorized dataset before running it.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data and credentials

Datasets, serialized models, cloud credentials, and access tokens are excluded.
