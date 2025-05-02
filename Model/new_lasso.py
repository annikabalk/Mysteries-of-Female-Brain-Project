import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, ElasticNet, LassoCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, f_regression
from scipy.stats import loguniform, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import joblib
from sklearn.ensemble import RandomForestRegressor

# Silence warnings
warnings.filterwarnings('ignore')

def main():
    script_start_time = time.time()
    print("Starting Enhanced Lasso model optimization using RandomizedSearchCV...")

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

    # --- 4. Enhanced Feature Selection ---
    print("\nPerforming enhanced feature selection...")
    
    # First, remove features with too many missing values (e.g., >50%)
    missing_threshold = 0.5
    missing_counts = X_orig_full.isnull().mean()
    cols_to_keep = missing_counts[missing_counts < missing_threshold].index.tolist()
    
    X_orig_filtered = X_orig_full[cols_to_keep]
    X_test_filtered = X_test_full[cols_to_keep]
    
    print(f"Removed {len(X_orig_full.columns) - len(cols_to_keep)} features with >{missing_threshold*100}% missing values")
    
    # Update column lists after removing high-missingness features
    numeric_cols = [col for col in numeric_cols_orig if col in cols_to_keep]
    categorical_cols = [col for col in categorical_cols_orig if col in cols_to_keep]
    
    # For numeric features, filter by variance (improved from original)
    if len(numeric_cols) > 0:
        # Only apply to numeric features with sufficient non-missing values
        X_numeric = X_orig_filtered[numeric_cols].copy()
        
        # Apply variance threshold to remove features with near-zero variance
        var_selector = VarianceThreshold(threshold=0.01)  # Adjusted threshold
        try:
            # Handle missing values temporarily for variance calculation
            X_numeric_imputed = SimpleImputer(strategy='median').fit_transform(X_numeric)
            var_selector.fit(X_numeric_imputed)
            selected_numeric_mask = var_selector.get_support()
            selected_numeric = [numeric_cols[i] for i in range(len(numeric_cols)) if selected_numeric_mask[i]]
            
            print(f"Removed {len(numeric_cols) - len(selected_numeric)} numeric features with near-zero variance")
            numeric_cols = selected_numeric
        except Exception as e:
            print(f"Variance-based selection skipped due to error: {e}")
    
    # Maximum number of features to keep
    max_features = 1000
    
    # If still too many features, use SelectKBest with f_regression
    total_features = len(numeric_cols) + len(categorical_cols)
    if total_features > max_features:
        # Prefer a more informative feature selection than just variance
        # Prepare a simple pipeline to impute and handle categorical features
        print(f"\nStill too many features ({total_features}). Using SelectKBest with f_regression...")
        
        # Process data for feature selection
        preprocessor_for_selection = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), numeric_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_cols)
            ],
            remainder='drop'
        )
        
        # Apply preprocessing for selection
        X_preprocessed = preprocessor_for_selection.fit_transform(X_orig_filtered)
        
        # Use SelectKBest with f_regression
        selector = SelectKBest(f_regression, k=min(max_features, X_preprocessed.shape[1]))
        selector.fit(X_preprocessed, y_orig)
        
        # Get feature importance scores for numeric features
        # This is tricky because of the preprocessing - need to extract feature names
        feature_names = numeric_cols.copy()
        
        # Get categorical feature names after one-hot encoding
        if len(categorical_cols) > 0:
            cat_encoder = preprocessor_for_selection.named_transformers_['cat'].named_steps['onehot']
            cat_features = []
            for i, col in enumerate(categorical_cols):
                encoded_features = [f"{col}_{val}" for val in cat_encoder.categories_[i]]
                cat_features.extend(encoded_features)
            feature_names.extend(cat_features)
        
        # Get scores and sort features by importance
        scores = selector.scores_
        if len(scores) < len(feature_names):
            print("Warning: Feature dimension mismatch. Using simpler selection method.")
            # Fall back to simpler method based on original code
            numeric_variances = X_orig_filtered[numeric_cols].var(axis=0).fillna(0)
            num_numeric_to_keep = max(0, max_features - len(categorical_cols))
            selected_numeric = numeric_variances.nlargest(num_numeric_to_keep).index.tolist()
            selected_features = selected_numeric + categorical_cols
            if len(selected_features) > max_features:
                selected_features = selected_features[:max_features]
        else:
            # Get top features based on scores
            feature_score_pairs = list(zip(feature_names, scores))
            feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_features = [pair[0] for pair in feature_score_pairs[:max_features]]
            
            # Extract original feature names from encoded features
            final_selected_features = []
            for feat in selected_features:
                if '_' in feat and any(feat.startswith(cat_col + '_') for cat_col in categorical_cols):
                    # Extract original categorical column name
                    cat_col = next(col for col in categorical_cols if feat.startswith(col + '_'))
                    if cat_col not in final_selected_features:
                        final_selected_features.append(cat_col)
                else:
                    # Regular numeric feature
                    if feat in numeric_cols:
                        final_selected_features.append(feat)
            
            selected_features = final_selected_features
        
        print(f"Selected {len(selected_features)} features based on importance scores")
        
        # Update feature and column lists
        feature_cols = [col for col in selected_features if col in cols_to_keep]
        numeric_cols = [col for col in numeric_cols if col in feature_cols]
        categorical_cols = [col for col in categorical_cols if col in feature_cols]
        
        # Filter datasets
        X_orig = X_orig_filtered[feature_cols]
        X_test = X_test_filtered[feature_cols]
    else:
        print("\nFeature count below threshold, using all filtered features.")
        feature_cols = numeric_cols + categorical_cols
        X_orig = X_orig_filtered[feature_cols]
        X_test = X_test_filtered[feature_cols]
    
    print(f"Final feature set: {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    
    # --- 5. Build Enhanced Preprocessing Pipeline ---
    print("\nBuilding enhanced preprocessing pipeline...")
    
    # Advanced numeric preprocessing with PowerTransformer for better handling of skewed data
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  # KNN imputation often better than median
        ('power_transform', PowerTransformer(method='yeo-johnson', standardize=True))  # Better than just StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))  # drop='if_binary' reduces dimensionality
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # --- 6. Define Full Pipeline with Lasso and ElasticNet ---
    print("Defining full pipeline with enhanced models...")
    
    # Create multiple pipeline options for model selection
    pipelines = {
        'lasso': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('lasso', Lasso(random_state=123, max_iter=3000))  # Increased max_iter for better convergence
        ]),
        'elastic_net': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('elastic_net', ElasticNet(random_state=123, max_iter=3000))  # ElasticNet for comparison
        ])
    }
    
    # Choose pipeline to optimize - we'll use both and compare
    print("\nWe'll optimize both Lasso and ElasticNet models and select the best")
    
    # --- 7. Define Enhanced Hyperparameter Search Space ---
    print("Defining enhanced hyperparameter search spaces...")
    
    # Lasso parameters
    lasso_param_dist = {
        'lasso__alpha': loguniform(1e-5, 1e1),  # Wider range
        'lasso__fit_intercept': [True, False],
        'lasso__positive': [True, False],  # Force coefficients to be positive (domain knowledge dependent)
        'lasso__selection': ['cyclic', 'random']  # Algorithm to update coordinates
    }
    
    # ElasticNet parameters
    elastic_net_param_dist = {
        'elastic_net__alpha': loguniform(1e-5, 1e1),
        'elastic_net__l1_ratio': uniform(0, 1),  # Mix between L1 and L2 penalties
        'elastic_net__fit_intercept': [True, False],
        'elastic_net__selection': ['cyclic', 'random']
    }

    # --- 8. Setup Enhanced Cross-Validation and Search ---
    print("Setting up enhanced CV and search...")
    
    N_SPLITS = 5
    N_ITER_SEARCH = 40  # Increased from 30 to 40 for better exploration
    
    # Stratified K-Fold for better distribution across folds
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=123)
    
    # Dictionary to store search results
    search_results = {}
    
    # --- 9. Run Hyperparameter Search for Both Models ---
    for model_name, pipeline in pipelines.items():
        print(f"\nStarting Randomized Search CV for {model_name.upper()}...")
        
        # Select appropriate parameter distribution
        if model_name == 'lasso':
            param_dist = lasso_param_dist
        else:  # elastic_net
            param_dist = elastic_net_param_dist
        
        # Configure search
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            cv=cv,
            scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],  # Multiple metrics
            refit='r2',  # Still optimize for R2
            n_jobs=-1,
            verbose=1,
            random_state=123,
            return_train_score=True  # Helpful for diagnosing overfitting
        )
        
        # Run search
        search_start_time = time.time()
        random_search.fit(X_orig, y_orig)
        search_end_time = time.time()
        
        print(f"{model_name.upper()} search completed in {(search_end_time - search_start_time):.2f} seconds.")
        
        # Store results
        search_results[model_name] = {
            'search': random_search,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_
        }
        
        # Report best results
        print("\n" + "="*50)
        print(f"{model_name.upper()} Hyperparameter Search Results:")
        print("="*50)
        print(f"Best parameters found:\n{random_search.best_params_}")
        print(f"\nBest cross-validation R2 score: {random_search.best_score_:.4f}")
        print(f"Best cross-validation RMSE: {np.sqrt(-random_search.cv_results_['mean_test_neg_mean_squared_error'][random_search.best_index_]):.4f}")
        print("="*50)
        
        # Save the model
        joblib.dump(random_search.best_estimator_, f'optimized_{model_name}_model.pkl')
    
    # Find the best overall model
    best_model_name = max(search_results, key=lambda k: search_results[k]['best_score'])
    best_model = search_results[best_model_name]['best_estimator']
    best_cv_r2 = search_results[best_model_name]['best_score']
    
    print(f"\nBest overall model: {best_model_name.upper()} with R2 score: {best_cv_r2:.4f}")
    
    # --- 10. Recreate Validation Split and Evaluate Final Model ---
    print(f"\nRecreating validation split for final {best_model_name.upper()} model evaluation...")
    
    temp_train_df, val_data = train_test_split(
        train_data, 
        test_size=0.2,
        random_state=123
    )
    
    # Extract features and target
    X_val = val_data[feature_cols]
    y_val = val_data[target_col].values
    
    print(f"Evaluating the best {best_model_name.upper()} model on validation set ({X_val.shape[0]} samples)...")
    
    # Predict using the best model
    val_preds = best_model.predict(X_val)
    
    # Calculate validation metrics
    val_r2 = r2_score(y_val, val_preds)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)
    
    print("\n" + "="*50)
    print(f"Final {best_model_name.upper()} Model Performance on Validation Set:")
    print("="*50)
    print(f"  R² Score: {val_r2:.4f}")
    print(f"  RMSE:     {val_rmse:.4f}")
    print(f"  MAE:      {val_mae:.4f}")
    print("="*50)

    # --- 11. Feature Importance Analysis ---
    if best_model_name == 'lasso':
        lasso_model = best_model.named_steps['lasso']
        # Get feature names after preprocessing
        feature_names = []
        
        # Get numeric feature names (unchanged)
        if len(numeric_cols) > 0:
            feature_names.extend(numeric_cols)
        
        # Get categorical feature names after one-hot encoding
        if len(categorical_cols) > 0:
            cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            for i, col in enumerate(categorical_cols):
                encoded_features = [f"{col}_{val}" for val in cat_encoder.categories_[i]]
                feature_names.extend(encoded_features)
        
        # Get coefficients
        coefficients = lasso_model.coef_
        
        # If dimensions don't match, skip feature importance
        if len(coefficients) == len(feature_names):
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(coefficients)
            }).sort_values('Importance', ascending=False)
            
            # Save top N features
            top_n = min(30, len(feature_importance))
            top_features = feature_importance.head(top_n)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top {top_n} Features by Importance ({best_model_name.upper()} Model)')
            plt.tight_layout()
            plt.savefig(f'{best_model_name}_feature_importance.png')
            plt.close()
            
            # Save to CSV
            feature_importance.to_csv(f'{best_model_name}_feature_importance.csv', index=False)
            print(f"Saved feature importance to {best_model_name}_feature_importance.csv")
    
    # --- 12. Save Validation Metrics ---
    print(f"\nSaving {best_model_name.upper()} validation metrics to file...")
    
    metrics_filename = f'optimized_{best_model_name}_validation_metrics.txt'
    with open(metrics_filename, 'w') as f:
        f.write(f"OPTIMIZED {best_model_name.upper()} MODEL - VALIDATION SET EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Cross-Validation R2 during search: {best_cv_r2:.4f}\n\n")
        f.write(f"Best Parameters Found ({best_model_name.upper()}):\n")
        for param, value in search_results[best_model_name]['best_params'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\nValidation Set Metrics (on recreated split):\n")
        f.write(f"  R² Score: {val_r2:.4f}\n")
        f.write(f"  RMSE:     {val_rmse:.4f}\n")
        f.write(f"  MAE:      {val_mae:.4f}\n")
    
    print(f"Saved validation metrics to '{metrics_filename}'")
    
    # --- 13. Save Test Set Predictions ---
    print(f"\nSaving test set predictions using {best_model_name.upper()} model...")
    
    test_preds = best_model.predict(X_test)
    test_results = pd.DataFrame({
        'participant_id': test_data['participant_id'],
        'age': test_preds
    })
    
    test_results_filename = f'test_predictions_{best_model_name}_optimized.csv'
    test_results.to_csv(test_results_filename, index=False)
    print(f"Saved test predictions to '{test_results_filename}'")
    
    # --- 14. Create Visualizations ---
    print(f"\nCreating visualizations using the {best_model_name.upper()} validation set results...")
    
    # Actual vs Predicted plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_val, val_preds, alpha=0.5)
    
    # Add regression line
    z = np.polyfit(y_val, val_preds, 1)
    p = np.poly1d(z)
    plt.plot(y_val, p(y_val), "r--", lw=2)
    
    # Add perfect prediction line
    min_val_plot = min(y_val.min(), val_preds.min())
    max_val_plot = max(y_val.max(), val_preds.max())
    plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'k--', lw=1)
    
    plt.title(f'Optimized {best_model_name.upper()}: Actual vs Predicted Age (Validation Set)\nRMSE={val_rmse:.2f}, R²={val_r2:.2f}')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'optimized_{best_model_name}_validation_actual_vs_predicted.png')
    plt.close()
    
    # Residuals plot (new)
    residuals = y_val - val_preds
    plt.figure(figsize=(10, 6))
    plt.scatter(val_preds, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Optimized {best_model_name.upper()}: Residuals Plot')
    plt.xlabel('Predicted Age')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'optimized_{best_model_name}_residuals_plot.png')
    plt.close()
    
    # Distribution plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_orig, label=f'Original Training Actual ({len(y_orig)})', fill=True, alpha=0.3, bw_adjust=0.5)
    sns.kdeplot(y_val, label=f'Validation Actual ({len(y_val)})', fill=True, alpha=0.3, color='green', bw_adjust=0.5)
    sns.kdeplot(val_preds, label=f'{best_model_name.upper()} Validation Predicted', fill=True, alpha=0.3, color='red', bw_adjust=0.5)
    
    plt.title(f'Optimized {best_model_name.upper()}: Age Distributions Comparison')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'optimized_{best_model_name}_validation_age_distributions.png')
    plt.close()
    
    # Error distribution plot (new)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Optimized {best_model_name.upper()}: Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'optimized_{best_model_name}_error_distribution.png')
    plt.close()
    
    # Report final results
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    print(f"\nOptimization and evaluation complete in {total_time:.2f} seconds.")
    print(f"FINAL {best_model_name.upper()} VALIDATION R² SCORE: {val_r2:.4f}")
    print(f"Models and results saved to disk. The best model was {best_model_name.upper()}.")

if __name__ == "__main__":
    main()