# Model Card: NYC Airbnb Rental Price Predictor

For additional information on model cards, see the Google Model Cards paper: https://arxiv.org/abs/1810.03993

## Model Details

- Developed as part of the Udacity Machine Learning DevOps "Build an ML Pipeline" project.
- The model is a **Random Forest Regressor** (scikit-learn `RandomForestRegressor`) wrapped in an sklearn inference pipeline.
- The full pipeline is orchestrated with **MLflow**, configured with **Hydra**, and tracked with **Weights & Biases**.
- The inference pipeline includes preprocessing: ordinal encoding for `room_type`, imputation + one-hot encoding for `neighbourhood_group`, zero-imputation for numeric columns, a date-delta feature engineered from `last_review`, and TF-IDF on the listing `name`.
- Key hyperparameters (defaults after tuning): `n_estimators=200`, `max_depth=50`, `max_features=0.5`, `min_samples_split=4`, `min_samples_leaf=3`, `random_state=42`.

## Intended Use

- **Primary use:** Estimate the nightly price of a short-term rental property in New York City based on features similar to comparable listings.
- **Intended users:** A property management company that receives new listing data in bulk on a regular cadence and needs the model retrained end-to-end with each new batch.
- **Out of scope:** This is a baseline model. It is not intended for high-stakes pricing decisions without human review, nor for cities/regions outside NYC.

## Training Data

- Source: a sample of the NYC Airbnb open dataset (`sample1.csv`), downloaded and versioned as a W&B artifact.
- Basic cleaning removes price outliers (kept only listings priced between $10 and $350), converts `last_review` to datetime, and filters out listings whose coordinates fall outside the NYC bounding box (longitude -74.25 to -73.50, latitude 40.5 to 41.2).
- The cleaned data is split into train/validation/test sets (20% test, 20% validation), stratified by `neighbourhood_group`, with `random_seed=42` for reproducibility.

## Evaluation Data

- The held-out **test set** (`test_data.csv`) produced by the split step, never seen during training or validation.

## Metrics

The model is evaluated using **Mean Absolute Error (MAE)** and the **R2 score**.

- Validation performance (best model): MAE ~ **34.18**, R2 ~ **0.55**.
- Test set performance: MAE ~ **33.85**, R2 ~ **0.56**.

The test MAE is comparable to the validation MAE, indicating the model generalizes well and is **not overfitting**.

## Ethical Considerations

- Airbnb listing data can reflect existing socioeconomic and geographic biases in the housing market. Predictions may reproduce those patterns, so the model should not be used as the sole basis for pricing that could affect housing access or fairness.
- The dataset contains host and listing identifiers; care should be taken not to expose personally identifiable information.

## Caveats and Recommendations

- The modeling here is deliberately kept simple to focus on the **MLOps** aspects (reproducible, reusable, end-to-end pipeline). Better predictive performance is achievable with more feature engineering and alternative models.
- Because the company receives new data regularly, the pipeline is designed to be retrained on new samples (e.g. `sample2.csv`) via a versioned release, ensuring reproducibility across retraining cycles.
- Future improvements could include richer EDA-driven feature engineering, trying gradient-boosted models, and expanding hyperparameter search.
