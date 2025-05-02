import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split # Added train_test_split back
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint # For parameter distributions (uniform not used here but good to have)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time # To time the search

# Silence warnings
warnings.filterwarnings('ignore')

def main():
    start_time = time.time()
    print("Starting MLP model optimization using RandomizedSearchCV...")

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
    # Keep participant_id if needed for saving test results later
    metadata_cols = ["participant_id", "sex", "bmi"] # Keep relevant metadata, age is target
    target_col = "age"

    # Ensure target exists in training data
    if target_col not in train_data.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return
    
    # Define potential feature columns (exclude target and specified metadata)
    potential_feature_cols = [col for col in train_data.columns if col not in metadata_cols + [target_col]]
    
    # Separate features and target from the *entire original training data* for CV search
    X_orig = train_data[potential_feature_cols]
    y_orig = train_data[target_col].values

    # Prepare test features (target assumed missing based on previous runs)
    # Use potential_feature_cols initially, will be filtered later if selection occurs
    X_test_orig = test_data[potential_feature_cols] 

    print(f"\nOriginal training set size: {train_data.shape[0]} rows")
    print(f"Test set size: {test_data.shape[0]} rows")
    print(f"Using {len(potential_feature_cols)} potential feature columns initially.")

    # --- 3. Identify Column Types from Original Training Data ---
    print("\nIdentifying column types from original features...")
    numeric_cols_orig = X_orig.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_orig = X_orig.select_dtypes(exclude=np.number).columns.tolist()

    print(f"Identified {len(numeric_cols_orig)} potential numeric feature columns")
    print(f"Identified {len(categorical_cols_orig)} potential categorical feature columns")

    # --- 4. Feature Selection (Optional - Example using Variance Threshold) ---
    # Applied before defining the final feature set used in the pipeline
    
    feature_cols = potential_feature_cols # Start with all potential features
    numeric_cols = numeric_cols_orig[:]   # Copy lists
    categorical_cols = categorical_cols_orig[:] 
    
    max_features = 1000
    if len(feature_cols) > max_features:
        print(f"\nPerforming feature selection: Selecting top {max_features} features based on variance...")

        # Calculate variance only on numeric columns of the original training data
        numeric_variances = X_orig[numeric_cols].var(axis=0).fillna(0) # Fill NaN variances with 0

        # Select top k numeric features, ensuring we don't exceed max_features even with categoricals
        num_numeric_to_keep = max(0, max_features - len(categorical_cols)) 
        top_numeric_features = numeric_variances.nlargest(num_numeric_to_keep).index.tolist()
        
        # Combine selected numeric features with all categorical features
        selected_features = top_numeric_features + categorical_cols
        
        # If still too many (e.g., many categoricals), truncate (could prioritize variance more here if needed)
        if len(selected_features) > max_features:
             selected_features = selected_features[:max_features] # Simple truncation

        print(f"Selected {len(selected_features)} features.")

        # Update feature lists and dataframes for pipeline definition and test set filtering
        feature_cols = selected_features # This is the final list of features used
        numeric_cols = [col for col in numeric_cols_orig if col in feature_cols]
        categorical_cols = [col for col in categorical_cols_orig if col in feature_cols]
        
        # Filter X_orig and X_test_orig TO MATCH the selected features for the pipeline
        X_orig = X_orig[feature_cols]
        X_test = X_test_orig[feature_cols] # Final test set features
        
        print(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features after selection.")
    else:
        print("\nSkipping feature selection (feature count below threshold).")
        X_test = X_test_orig # Use all original features if none selected
        # numeric_cols and categorical_cols remain as originally identified

    # --- 5. Build Preprocessing Pipeline ---
    print("\nBuilding preprocessing pipeline...")
    # Use the final numeric_cols and categorical_cols lists AFTER selection
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
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
        remainder='drop' # Drop any columns not explicitly handled (e.g., if selection logic changes)
    )

    # --- 6. Define Full Pipeline with MLP ---
    print("Defining full pipeline (Preprocessor + MLP)...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('mlp', MLPRegressor(random_state=123,
                             max_iter=300,
                             early_stopping=True,
                             validation_fraction=0.1,
                             n_iter_no_change=15))
    ])

    # --- 7. Define Hyperparameter Search Space ---
    print("Defining hyperparameter search space...")
    param_dist = {
        # Use pipeline syntax: 'stepname__parametername'
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50), (150, 75, 30), (200,100,50)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam'],
        'mlp__alpha': loguniform(1e-5, 1e-1),
        'mlp__learning_rate_init': loguniform(1e-4, 1e-2),
        'mlp__batch_size': [32, 64, 128],
        'mlp__learning_rate': ['constant', 'adaptive'],
    }

    # --- 8. Setup Cross-Validation and Randomized Search ---
    print("Setting up KFold Cross-Validation and Randomized Search...")
    N_SPLITS = 5
    N_ITER_SEARCH = 30 # Adjust as needed

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=123)

    # Optimize for R2 score now
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='r2', # <<< SCORE FOR R2
        n_jobs=-1,
        verbose=2,
        random_state=123,
        refit=True # Refit the best estimator on the whole training set (X_orig, y_orig)
    )

    # --- 9. Run Hyperparameter Search on Original Training Data ---
    print(f"\nStarting Randomized Search CV ({N_ITER_SEARCH} iterations, {N_SPLITS}-fold CV) optimizing for R2...")
    search_start_time = time.time()
    # Fit the search on the training data (X_orig has selected features now)
    random_search.fit(X_orig, y_orig)
    search_end_time = time.time()
    print(f"Randomized Search CV completed in {(search_end_time - search_start_time):.2f} seconds.")

    # --- 10. Report Best Results from Search ---
    print("\n" + "="*50)
    print("Hyperparameter Search Results:")
    print("="*50)
    print(f"Best parameters found:\n{random_search.best_params_}")
    best_cv_r2 = random_search.best_score_ # Score is directly R2 now
    print(f"\nBest cross-validation R2 score: {best_cv_r2:.4f}")
    print("="*50)

    # Get the best pipeline (already refitted on the entire X_orig, y_orig)
    best_pipeline = random_search.best_estimator_

    # --- 11. Recreate Validation Split and Evaluate Final Model ---
    # Now evaluate the single best model found on the held-out validation set
    print("\nRecreating validation split for final model evaluation...")
    
    # Re-split the *original* train_data to get the consistent validation set
    # Use the full train_data dataframe loaded at the start
    temp_train_df, val_data = train_test_split(
        train_data, # Use the original full training dataframe
        test_size=0.2,
        random_state=123 # Use the SAME random_state as your very first script
    )

    # Extract features (using the final 'feature_cols' list) and target for the validation set
    X_val = val_data[feature_cols]
    y_val = val_data[target_col].values

    print(f"Evaluating the best model on the recreated validation set ({X_val.shape[0]} samples)...")

    # Predict using the best_pipeline (it includes preprocessing)
    val_preds = best_pipeline.predict(X_val)

    # Calculate validation metrics
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)

    print("\n" + "="*50)
    print("Final Model Performance on Recreated Validation Set:")
    print("="*50)
    print(f"  R² Score: {val_r2:.4f}") # <<< R² Score on validation set
    print(f"  RMSE:     {val_rmse:.4f}")
    print(f"  MAE:      {val_mae:.4f}")
    print("="*50)

    # --- 12. Save Validation Metrics ---
    print("\nSaving validation metrics to file...")
    metrics_filename = 'mlp_optimized_validation_metrics.txt'
    with open(metrics_filename, 'w') as f:
        f.write("OPTIMIZED MLP MODEL - VALIDATION SET EVALUATION METRICS\n")
        f.write("="*55 + "\n\n")
        f.write(f"Best Cross-Validation R2 during search: {best_cv_r2:.4f}\n\n") # Report best CV R2
        f.write("Best Parameters Found:\n")
        for param, value in random_search.best_params_.items():
             f.write(f"  {param}: {value}\n")
        f.write("\nValidation Set Metrics (on recreated split):\n") # Clarify it's validation metrics
        f.write(f"  R² Score: {val_r2:.4f}\n")
        f.write(f"  RMSE:     {val_rmse:.4f}\n")
        f.write(f"  MAE:      {val_mae:.4f}\n")
    print(f"Saved validation metrics to '{metrics_filename}'")

    # --- 13. Save Test Set Predictions (using the final best model) ---
    print("\nSaving test set predictions (target labels assumed missing)...")
    # Use the final X_test which has features filtered according to selection
    test_preds = best_pipeline.predict(X_test)
    test_results = pd.DataFrame({
        'participant_id': test_data['participant_id'], # Assumes participant_id exists
        'predicted_age': test_preds
    })
    test_results_filename = 'test_predictions_mlp_optimized.csv'
    test_results.to_csv(test_results_filename, index=False)
    print(f"Saved test predictions to '{test_results_filename}'")

    # --- 14. Create Visualizations (using Validation Set results) ---
    print("\nCreating visualizations using the validation set results...")

    # Actual vs Predicted plot for validation set
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, val_preds, alpha=0.5)
    min_val_plot = min(y_val.min(), val_preds.min())
    max_val_plot = max(y_val.max(), val_preds.max())
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2)
    plt.title(f'Optimized MLP: Actual vs Predicted Age (Validation Set)\nRMSE={val_rmse:.2f}, R²={val_r2:.2f}')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_validation_actual_vs_predicted.png')
    plt.close()
    print("Saved: optimized_validation_actual_vs_predicted.png")

    # Distribution plot comparing original training data, validation actuals, and validation predictions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_orig, label=f'Original Training Actual ({len(y_orig)})', fill=True, alpha=0.3, bw_adjust=0.5)
    sns.kdeplot(y_val, label=f'Validation Actual ({len(y_val)})', fill=True, alpha=0.3, color='green', bw_adjust=0.5)
    sns.kdeplot(val_preds, label=f'Validation Predicted ({len(val_preds)})', fill=True, alpha=0.3, color='red', bw_adjust=0.5)
    plt.title('Optimized Model: Age Distributions Comparison')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_validation_age_distributions.png')
    plt.close()
    print("Saved: optimized_validation_age_distributions.png")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nOptimization and evaluation complete in {total_time:.2f} seconds.")
    # Report final R² score from validation set
    print(f"FINAL VALIDATION R² SCORE: {val_r2:.4f}")

if __name__ == "__main__":
    main()