import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso # <-- Import Lasso
# from sklearn.neural_network import MLPRegressor # <-- Removed MLP
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform # For parameter distributions
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time # To time the search

# Silence warnings
warnings.filterwarnings('ignore')

def main():
    script_start_time = time.time()
    print("Starting Lasso model optimization using RandomizedSearchCV...")

    # --- 1. Load Data ---
    print("\nLoading data...")
    try:
        # Load the original training data (needed for recreating val split)
        train_data = pd.read_csv("processed_train_data.csv")
        # Load test data (for final predictions - labels assumed missing)
        test_data = pd.read_csv("processed_test_data.csv")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'processed_train_data.csv' and 'processed_test_data.csv' are in the correct directory.")
        return

    # --- 2. Identify Features, Target, and Metadata ---
    metadata_cols = ["participant_id", "sex", "bmi"]
    target_col = "age"

    if target_col not in train_data.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return

    potential_feature_cols = [col for col in train_data.columns if col not in metadata_cols + [target_col]]
    
    X_orig_full = train_data[potential_feature_cols] # Keep full original features for selection step
    y_orig = train_data[target_col].values
    X_test_full = test_data[potential_feature_cols] # Keep full original test features

    print(f"\nOriginal training set size: {train_data.shape[0]} rows")
    print(f"Test set size: {test_data.shape[0]} rows")
    print(f"Using {len(potential_feature_cols)} potential feature columns initially.")

    # --- 3. Identify Column Types from Original Training Data ---
    print("\nIdentifying column types from original features...")
    numeric_cols_orig = X_orig_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_orig = X_orig_full.select_dtypes(exclude=np.number).columns.tolist()

    print(f"Identified {len(numeric_cols_orig)} potential numeric feature columns")
    print(f"Identified {len(categorical_cols_orig)} potential categorical feature columns")

    # --- 4. Feature Selection (Optional - Example using Variance Threshold) ---
    feature_cols = potential_feature_cols # Start with all
    numeric_cols = numeric_cols_orig[:]   # Copy lists
    categorical_cols = categorical_cols_orig[:]
    
    X_orig = X_orig_full # Start with original features before potentially filtering
    X_test = X_test_full

    max_features = 1000
    if len(feature_cols) > max_features:
        print(f"\nPerforming feature selection: Selecting top {max_features} features based on variance...")
        numeric_variances = X_orig_full[numeric_cols].var(axis=0).fillna(0)
        num_numeric_to_keep = max(0, max_features - len(categorical_cols))
        top_numeric_features = numeric_variances.nlargest(num_numeric_to_keep).index.tolist()
        selected_features = top_numeric_features + categorical_cols
        if len(selected_features) > max_features:
             selected_features = selected_features[:max_features]

        print(f"Selected {len(selected_features)} features.")
        feature_cols = selected_features
        numeric_cols = [col for col in numeric_cols_orig if col in feature_cols]
        categorical_cols = [col for col in categorical_cols_orig if col in feature_cols]
        
        # Filter datasets to use only selected features from here on
        X_orig = X_orig_full[feature_cols]
        X_test = X_test_full[feature_cols]

        print(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features after selection.")
    else:
        print("\nSkipping feature selection (feature count below threshold).")
        # X_orig, X_test, numeric_cols, categorical_cols remain as identified initially

    # --- 5. Build Preprocessing Pipeline ---
    print("\nBuilding preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Scaling is important for Lasso
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # --- 6. Define Full Pipeline with Lasso ---
    print("Defining full pipeline (Preprocessor + Lasso)...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('lasso', Lasso(random_state=123, max_iter=2000)) # Increased max_iter for Lasso
    ])

    # --- 7. Define Hyperparameter Search Space for Lasso ---
    print("Defining hyperparameter search space for Lasso...")
    # Parameters specific to Lasso, using the 'lasso__' prefix
    param_dist = {
        'lasso__alpha': loguniform(1e-4, 1e1) # Tune alpha over a wide range (log scale is often good)
        # Can add 'lasso__fit_intercept': [True, False] if desired
    }

    # --- 8. Setup Cross-Validation and Randomized Search ---
    print("Setting up KFold Cross-Validation and Randomized Search for Lasso...")
    N_SPLITS = 5
    N_ITER_SEARCH = 30 # Number of parameter settings to try for Lasso

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=123)

    # Using 'r2' scoring as requested previously
    random_search = RandomizedSearchCV(
        full_pipeline, # Use the pipeline with Lasso
        param_distributions=param_dist, # Use the Lasso param_dist
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='r2', # Optimize for R2 score
        n_jobs=-1,
        verbose=2,
        random_state=123,
        refit=True # Refit the best estimator on the whole training set
    )

    # --- 9. Run Hyperparameter Search ---
    print(f"\nStarting Randomized Search CV ({N_ITER_SEARCH} iterations, {N_SPLITS}-fold CV) for Lasso...")
    search_start_time = time.time()
    # Fit on the potentially filtered training data (X_orig, y_orig)
    random_search.fit(X_orig, y_orig)
    search_end_time = time.time()
    print(f"Randomized Search CV for Lasso completed in {(search_end_time - search_start_time):.2f} seconds.")

    # --- 10. Report Best Results from Search ---
    print("\n" + "="*50)
    print("Lasso Hyperparameter Search Results:") # <-- Updated Title
    print("="*50)
    print(f"Best parameters found:\n{random_search.best_params_}")
    best_cv_r2 = random_search.best_score_ # Score is R2
    print(f"\nBest cross-validation R2 score: {best_cv_r2:.4f}")
    print("="*50)

    # Get the best pipeline (already refitted)
    best_pipeline = random_search.best_estimator_

    # --- 11. Recreate Validation Split and Evaluate Final Lasso Model ---
    print("\nRecreating validation split for final Lasso model evaluation...") # <-- Updated Title

    temp_train_df, val_data = train_test_split(
        train_data, # Use the original full training dataframe
        test_size=0.2,
        random_state=123 # Use the SAME random_state
    )

    # Extract features (using the final 'feature_cols' list after selection) and target
    X_val = val_data[feature_cols]
    y_val = val_data[target_col].values

    print(f"Evaluating the best Lasso model on the recreated validation set ({X_val.shape[0]} samples)...") # <-- Updated Title

    # Predict using the best_pipeline (includes preprocessing + best Lasso)
    val_preds = best_pipeline.predict(X_val)

    # Calculate validation metrics
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)

    print("\n" + "="*50)
    print("Final Lasso Model Performance on Recreated Validation Set:") # <-- Updated Title
    print("="*50)
    print(f"  R² Score: {val_r2:.4f}")
    print(f"  RMSE:     {val_rmse:.4f}")
    print(f"  MAE:      {val_mae:.4f}")
    print("="*50)

    # --- 12. Save Validation Metrics for Lasso ---
    print("\nSaving Lasso validation metrics to file...") # <-- Updated Title
    metrics_filename = 'lasso_optimized_validation_metrics.txt' # <-- New Filename
    with open(metrics_filename, 'w') as f:
        f.write("OPTIMIZED LASSO MODEL - VALIDATION SET EVALUATION METRICS\n") # <-- Updated Title
        f.write("="*55 + "\n\n")
        f.write(f"Best Cross-Validation R2 during search: {best_cv_r2:.4f}\n\n")
        f.write("Best Parameters Found (Lasso):\n") # <-- Updated Title
        for param, value in random_search.best_params_.items():
             f.write(f"  {param}: {value}\n")
        f.write("\nValidation Set Metrics (on recreated split):\n")
        f.write(f"  R² Score: {val_r2:.4f}\n")
        f.write(f"  RMSE:     {val_rmse:.4f}\n")
        f.write(f"  MAE:      {val_mae:.4f}\n")
    print(f"Saved validation metrics to '{metrics_filename}'")

    # --- 13. Save Test Set Predictions using Lasso Model ---
    print("\nSaving test set predictions using Lasso model...") # <-- Updated Title
    # Use the final X_test which has features filtered according to selection
    test_preds = best_pipeline.predict(X_test)
    test_results = pd.DataFrame({
        'participant_id': test_data['participant_id'],
        'age': test_preds # Use 'age' as requested column name
    })
    test_results_filename = 'test_predictions_lasso_optimized.csv' # <-- New Filename
    test_results.to_csv(test_results_filename, index=False)
    print(f"Saved test predictions to '{test_results_filename}'")

    # --- 14. Create Visualizations for Lasso Model ---
    print("\nCreating visualizations using the Lasso validation set results...") # <-- Updated Title

    # Actual vs Predicted plot for validation set
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, val_preds, alpha=0.5)
    min_val_plot = min(y_val.min(), val_preds.min())
    max_val_plot = max(y_val.max(), val_preds.max())
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2)
    # Updated plot title for Lasso
    plt.title(f'Optimized Lasso: Actual vs Predicted Age (Validation Set)\nRMSE={val_rmse:.2f}, R²={val_r2:.2f}')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_lasso_validation_actual_vs_predicted.png') # <-- New Filename
    plt.close()
    print("Saved: optimized_lasso_validation_actual_vs_predicted.png")

    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_orig, label=f'Original Training Actual ({len(y_orig)})', fill=True, alpha=0.3, bw_adjust=0.5)
    sns.kdeplot(y_val, label=f'Validation Actual ({len(y_val)})', fill=True, alpha=0.3, color='green', bw_adjust=0.5)
    sns.kdeplot(val_preds, label=f'Lasso Validation Predicted ({len(val_preds)})', fill=True, alpha=0.3, color='red', bw_adjust=0.5) # <-- Updated Label
    # Updated plot title for Lasso
    plt.title('Optimized Lasso: Age Distributions Comparison')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_lasso_validation_age_distributions.png') # <-- New Filename
    plt.close()
    print("Saved: optimized_lasso_validation_age_distributions.png")

    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    print(f"\nLasso optimization and evaluation complete in {total_time:.2f} seconds.") # <-- Updated print
    # Report final R² score from validation set
    print(f"FINAL LASSO VALIDATION R² SCORE: {val_r2:.4f}") # <-- Updated print

if __name__ == "__main__":
    main()