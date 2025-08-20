We applied a modular cleaning pipeline to ensure reproducibility and clarity:

Fill missing values with median:
-Median chosen over mean to reduce sensitivity to outliers.
-Applied only to numeric columns

Drop highly missing columns:
-Any column with more than 50% missing values was dropped.
-Assumption: too much missingness makes imputation unreliable.

Normalize numeric columns:
-Min-Max scaling applied to bring all numeric features to [0, 1].
-Assumption: improves comparability across features and prepares data for ML models that are scale-sensitive.
