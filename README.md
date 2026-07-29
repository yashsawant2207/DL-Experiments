# Hotel Booking Demand: Robust Data Processing Pipeline

## Overview
This repository contains a production-ready data preprocessing pipeline for the **Hotel Booking Demand** dataset. The primary objective of this architecture is to transform raw booking records into a mathematically sound feature matrix for predictive modeling (specifically, predicting `is_canceled`), while strictly enforcing boundaries to prevent data leakage and structural bias.

## Dataset
*   **Source:** Antonio, Almeida, and Nunes (2019), *Data in Brief*.
*   **Initial Shape:** 119,390 observations, 32 features.
*   **Target Variable:** `is_canceled` (Binary).

## Methodological Architecture

### 1. Data Cleansing & Leakage Prevention
Raw operational datasets inherently contain administrative noise and post-event variables. The following actions were taken prior to any statistical processing:
*   **Leakage Elimination:** Removed `reservation_status` and `reservation_status_date`. These variables perfectly correlate with the target and allow the model to bypass actual pattern recognition.
*   **Anomaly Purging:** Filtered out logically invalid observations, specifically rows recording zero total guests (`adults` + `children` + `babies` = 0) and zero total nights.

### 2. Structural Missing Value Handling
Missing values were diagnosed by their mechanism of missingness rather than being blindly imputed with statistical measures (which would introduce severe bias).
*   **Missing Not At Random (MNAR):** 
    *   `company` (94.3% missing): Dropped entirely due to extreme sparsity; lacks sufficient variance for modeling.
    *   `agent` (13.7% missing): Nulls indicate direct bookings. Handled structurally by filling with `0`.
*   **Missing At Random (MAR):**
    *   `country` (0.4% missing): Filled with `'Unknown'` to treat the missingness as a distinct category, avoiding mode-imputation leakage.
*   **Missing Completely At Random (MCAR):**
    *   `children` (0.003% missing): Filled with `0`.

### 3. Feature Engineering
Derived features were constructed to consolidate fragmented signals into continuous or binary metrics suitable for tree-based or gradient descent optimization:
*   **Volume Metrics:** Aggregated `total_guests` and `total_nights`.
*   **Behavioral Signals:** Created a binary `room_assignment_changed` flag to capture operational friction (reserved vs. assigned room).
*   **Temporal Mapping:** Converted string-based arrival months into numerical representations to establish continuous temporal relationships.

### 4. Stratified Data Splitting
To maintain the integrity of model evaluation metrics (precision, recall), the dataset was split (80/20) using **stratified sampling**. This guarantees the ~37% cancellation rate is identical across both the training and testing sets, preventing skewed distribution shifts.

### 5. `scikit-learn` Pipeline Construction
Transformations were strictly confined within a `ColumnTransformer` and fitted **only** on the training set to prevent test data from influencing the statistical parameters of the transformations.

*   **Numeric Features:** Processed via `SimpleImputer(strategy='median')` -> `StandardScaler()`.
*   **Low-Cardinality Categoricals:** Processed via `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(handle_unknown='ignore')`.
*   **High-Cardinality Categoricals (`country`):** Processed via `TargetEncoder(target_type='binary')`. This replaces 177 sparse categorical columns with a single, dense probability vector, avoiding the curse of dimensionality.

## Generated Artifacts

Executing the pipeline yields three primary artifacts:

1.  **`processed_training_data.csv`**: The fitted and transformed training matrix, including the re-appended target variable.
2.  **`processed_testing_data.csv`**: The rigorously transformed testing matrix, modified solely by the parameters learned from the training data.
3.  **`hotel_booking_preprocessor.joblib`**: The serialized `ColumnTransformer` object. 
