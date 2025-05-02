import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import joblib

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
    metadata_cols = ["participant_id", "sex", "bmi"]
    target_col = "age"

    if target_col not in train_data.columns:
        print(f"Error: Target column '{target_col}' not found in training data.")
        return
    
    potential_feature_cols = [col for col in train_data.columns if col not in metadata_cols + [target_col]]
    
    X_orig_full = train_data[potential_feature_cols]
    y_orig = train_data[target_col].values
    X_test_full = test_data[potential_feature_cols]

    print(f"\nOriginal training set size: {train_data.shape[0]} rows")
    print(f"Test set size: {test_data.shape[0]} rows")
    print(f"Using {len(potential_feature_cols)} potential feature columns initially.")

    # --- 3. Identify Column Types from Original Training Data ---
    print("\nIdentifying column types from original features...")
    numeric_cols_orig = X_orig_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_orig = X_orig_full.select_dtypes(exclude=np.number).columns.tolist()

    print(f"Identified {len(numeric_cols_orig)} potential numeric feature columns")
    print(f"Identified {len(categorical_cols_orig)} potential categorical feature columns")

    # --- 4. Feature Selection ---
    # Keep feature selection logic similar to original - avoiding too many changes
    feature_cols = potential_feature_cols  # Start with all potential features
    numeric_cols = numeric_cols_orig[:]    # Copy lists
    categorical_cols = categorical_cols_orig[:]
    
    # Keep original variance-based selection approach
    max_features = 1000
    if len(feature_cols) > max_features:
        print(f"\nPerforming feature selection: Selecting top {max_features} features based on variance...")

        # Calculate variance only on numeric columns
        numeric_variances = X_orig_full[numeric_cols].var(axis=0).fillna(0)

        # Select top k numeric features
        num_numeric_to_keep = max(0, max_features - len(categorical_cols))
        top_numeric_features = numeric_variances.nlargest(num_numeric_to_keep).index.tolist()
        
        # Combine selected numeric features with all categorical features
        selected_features = top_numeric_features + categorical_cols
        
        if len(selected_features) > max_features:
             selected_features = selected_features[:max_features]

        print(f"Selected {len(selected_features)} features.")

        # Update feature lists
        feature_cols = selected_features
        numeric_cols = [col for col in numeric_cols_orig if col in feature_cols]
        categorical_cols = [col for col in categorical_cols_orig if col in feature_cols]
        
        # Filter X_orig_full and X_test_full
        X_orig = X_orig_full[feature_cols]
        X_test = X_test_full[feature_cols]
        
        print(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features after selection.")
    else:
        print("\nSkipping feature selection (feature count below threshold).")
        X_orig = X_orig_full
        X_test = X_test_full

    # --- 5. Build Preprocessing Pipeline ---
    print("\nBuilding preprocessing pipeline...")
    # Using more robust preprocessing but keeping it simple
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # RobustScaler handles outliers better
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

    # --- 6. Define Full Pipeline with MLP ---
    print("Defining full pipeline (Preprocessor + MLP)...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('mlp', MLPRegressor(
            random_state=123,
            max_iter=300,  # Keeping the same as original
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        ))
    ])

    # --- 7. Define Hyperparameter Search Space ---
    print("Defining hyperparameter search space...")
    # Keep the core parameters but refine the ranges
    param_dist = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (150,), (50, 25), (100, 50), (150, 75)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__solver': ['adam'],
        'mlp__alpha': loguniform(1e-5, 1e-3),  # Narrower range to avoid extreme regularization
        'mlp__learning_rate_init': loguniform(1e-4, 1e-2),
        'mlp__batch_size': [32, 64, 128],
        'mlp__learning_rate': ['constant', 'adaptive'],
    }

    # --- 8. Setup Cross-Validation and Randomized Search ---
    print("Setting up KFold Cross-Validation and Randomized Search...")
    N_SPLITS = 5
    N_ITER_SEARCH = 30  # Keep same as original

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=123)

    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1,  # Reduced verbosity
        random_state=123,
        refit=True
    )

    # --- 9. Run Hyperparameter Search ---
    print(f"\nStarting Randomized Search CV ({N_ITER_SEARCH} iterations, {N_SPLITS}-fold CV)...")
    search_start_time = time.time()
    random_search.fit(X_orig, y_orig)
    search_end_time = time.time()
    print(f"Randomized Search CV completed in {(search_end_time - search_start_time):.2f} seconds.")

    # --- 10. Report Best Results from Search ---
    print("\n" + "="*50)
    print("Hyperparameter Search Results:")
    print("="*50)
    print(f"Best parameters found:\n{random_search.best_params_}")
    best_cv_r2 = random_search.best_score_
    print(f"\nBest cross-validation R2 score: {best_cv_r2:.4f}")
    print("="*50)

    # Sanity check - warn if CV R2 score is negative
    if best_cv_r2 < 0:
        print("\nWARNING: Best CV R2 score is negative! The model might be performing poorly.")
        print("Continuing evaluation to confirm and diagnose issues...")

    # Get the best pipeline
    best_pipeline = random_search.best_estimator_
    
    # Optional: Save the best model
    joblib.dump(best_pipeline, 'best_mlp_model.pkl')
    print("Saved best model to best_mlp_model.pkl")

    # --- 11. Recreate Validation Split and Evaluate Final Model ---
    print("\nRecreating validation split for final model evaluation...")
    
    temp_train_df, val_data = train_test_split(
        train_data,
        test_size=0.2,
        random_state=123
    )
    
    X_val = val_data[feature_cols]
    y_val = val_data[target_col].values
    
    print(f"Evaluating the best model on the validation set ({X_val.shape[0]} samples)...")
    
    val_preds = best_pipeline.predict(X_val)
    
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)
    
    print("\n" + "="*50)
    print("Final Model Performance on Validation Set:")
    print("="*50)
    print(f"  R² Score: {val_r2:.4f}")
    print(f"  RMSE:     {val_rmse:.4f}")
    print(f"  MAE:      {val_mae:.4f}")
    print("="*50)
    
    # Additional sanity check for validation R2
    if val_r2 < 0:
        print("\nWARNING: Validation R2 score is negative! The model is performing worse than a baseline.")
        print("Diagnostic step: Checking training set R2 to detect overfitting...")
        
        # Check training performance
        train_preds = best_pipeline.predict(X_orig)
        train_r2 = r2_score(y_orig, train_preds)
        print(f"Training set R2: {train_r2:.4f}")
        
        if train_r2 > 0.1 and val_r2 < 0:
            print("Likely overfitting: Good training performance but poor validation performance.")
        elif train_r2 < 0.1:
            print("Model failed to learn: Poor performance on both training and validation data.")

    # --- 12. Save Validation Metrics ---
    print("\nSaving validation metrics to file...")
    
    metrics_filename = 'mlp_optimized_validation_metrics.txt'
    with open(metrics_filename, 'w') as f:
        f.write("OPTIMIZED MLP MODEL - VALIDATION SET EVALUATION METRICS\n")
        f.write("="*55 + "\n\n")
        f.write(f"Best Cross-Validation R2 during search: {best_cv_r2:.4f}\n\n")
        f.write("Best Parameters Found:\n")
        for param, value in random_search.best_params_.items():
             f.write(f"  {param}: {value}\n")
        f.write("\nValidation Set Metrics (on recreated split):\n")
        f.write(f"  R² Score: {val_r2:.4f}\n")
        f.write(f"  RMSE:     {val_rmse:.4f}\n")
        f.write(f"  MAE:      {val_mae:.4f}\n")
    
    print(f"Saved validation metrics to '{metrics_filename}'")
    
    # --- 13. Save Test Set Predictions ---
    print("\nSaving test set predictions...")
    
    test_preds = best_pipeline.predict(X_test)
    test_results = pd.DataFrame({
        'participant_id': test_data['participant_id'],
        'age': test_preds
    })
    
    test_results_filename = 'test_predictions_mlp_optimized.csv'
    test_results.to_csv(test_results_filename, index=False)
    print(f"Saved test predictions to '{test_results_filename}'")
    
    # --- 14. Create Visualizations ---
    print("\nCreating visualizations...")
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val, val_preds, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(y_val, val_preds, 1)
    p = np.poly1d(z)
    plt.plot(y_val, p(y_val), "r--", lw=2)
    
    # Add perfect prediction line
    min_val_plot = min(y_val.min(), val_preds.min())
    max_val_plot = max(y_val.max(), val_preds.max())
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', lw=1)
    
    plt.title(f'Optimized MLP: Actual vs Predicted Age (Validation Set)\nRMSE={val_rmse:.2f}, R²={val_r2:.2f}')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_mlp_validation_actual_vs_predicted.png')
    plt.close()
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_orig, label=f'Original Training Actual ({len(y_orig)})', fill=True, alpha=0.3, bw_adjust=0.5)
    sns.kdeplot(y_val, label=f'Validation Actual ({len(y_val)})', fill=True, alpha=0.3, color='green', bw_adjust=0.5)
    sns.kdeplot(val_preds, label=f'MLP Validation Predicted', fill=True, alpha=0.3, color='red', bw_adjust=0.5)
    
    plt.title('Optimized MLP: Age Distributions Comparison')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_mlp_validation_age_distributions.png')
    plt.close()
    
    # Residuals plot
    residuals = y_val - val_preds
    plt.figure(figsize=(10, 6))
    plt.scatter(val_preds, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Optimized MLP: Residuals Plot')
    plt.xlabel('Predicted Age')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('optimized_mlp_residuals_plot.png')
    plt.close()
    
    # Report final results
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nMLP optimization and evaluation complete in {total_time:.2f} seconds.")
    print(f"FINAL VALIDATION R² SCORE: {val_r2:.4f}")
    
    # Fallback recommendation if model still performs poorly
    if val_r2 < 0:
        print("\n" + "="*60)
        print("IMPORTANT: Model performance is still poor. Consider these alternatives:")
        print("1. Try simpler models (Linear Regression, Lasso, Ridge) as baselines")
        print("2. Review feature engineering - the current features may not be predictive")
        print("3. Decrease model complexity to reduce overfitting")
        print("4. Consider additional data preprocessing steps")
        print("5. Check for data quality issues or leakage in the validation split")
        print("="*60)

if __name__ == "__main__":
    main()