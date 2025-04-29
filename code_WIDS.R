setwd("C:/MSSP Final Portfolio")
### Since we are interested in fitting a model using training data lets only read in the training data for now

library(tidyverse)
library(vroom)
library(stringr)
library(broom)
library(factoextra)
library(caret)
library(tidymodels)
library(ranger)
library(vip)

# Function to extract upper triangle from a matrix (excluding diagonal)
extract_upper_triangle <- function(mat) {
  mat[upper.tri(mat, diag = FALSE)] %>% as.numeric()
}

process_connectomes <- function(folder_path) {
  tsv_files <- list.files(folder_path, pattern = "\\.tsv$", full.names = TRUE)
  
  map_df(tsv_files, ~ {
    mat <- as.matrix(vroom(.x, col_names = FALSE, delim = "\t"))
    full_id <- tools::file_path_sans_ext(basename(.x))  # Original filename without extension
    
    # Extract participant ID (characters 5-14 after "sub-")
    participant_id <- str_sub(full_id, start = 5, end = 16)  # Using stringr
    
    tibble(
      id = participant_id,  # Now contains just NDARAA306NT2 etc.
      full_filename = full_id,  # Optional: keep original for reference
      feature_vector = list(extract_upper_triangle(mat))
    )
  })
}

# Function to convert list of feature vectors to a dataframe
expand_features <- function(processed_data) {
  processed_data %>%
    mutate(feature_index = map(feature_vector, ~ seq_along(.x))) %>%
    unnest(c(feature_vector, feature_index)) %>%
    pivot_wider(
      names_from = feature_index,
      values_from = feature_vector,
      names_prefix = "feature_"
    )
}

# Main processing pipeline
process_dataset <- function(input_folder, output_file = NULL) {
  # Process the folder
  processed <- process_connectomes(input_folder)
  
  # Expand features into columns
  expanded <- expand_features(processed)
  
  # Save if output file specified
  if (!is.null(output_file)) {
    write_csv(expanded, output_file)
  }
  
  return(expanded)
}


# Run for training and test sets
train_path <- "train_tsv/train_tsv"
test_path <- "test_tsv/test_tsv"

train_df <- process_dataset(train_path, "train_processed.csv")
test_df <- process_dataset(test_path, "test_processed.csv")


#########################################################################
#########################################################################
########### PCA To reduce dimensions of data before combining with metadata

## Read it in as a csv and not a tibble.

train.csv <- read.csv("train_processed.csv")
test.csv <- read.csv("test_processed.csv")

## Use only features to perform PCAs
feature_matrix <- train.csv %>%
  select(starts_with("feature_")) %>%
  as.matrix()

# Scale data
scaled_data <- scale(feature_matrix, center = TRUE, scale = TRUE)


### Set seed for reproducibility
set.seed(100)


## Perform PCA
pca_result <- prcomp(scaled_data, retx = TRUE)

## Look at scree plot to determine number of components
fviz_eig(pca_result, addlabels = TRUE) + 
  ggtitle("Scree Plot - PCA Eigenvalues") +
  theme_minimal()

## Find the PC's that explain 90% of the variance
cumulative_var <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
n_components <- which(cumulative_var >= 0.9)[1]
## n_components there are 654
## Successfully reduced the data from 19k features to 654 features

pca_scores <- as_tibble(pca_result$x[, 1:n_components]) %>%
  mutate(participant_id = train.csv$participant_id)

## final reduced data set
reduced_data <- pca_scores %>%
  select(participant_id, starts_with("PC"))

### Load in metadata

meta_train <- read.csv("metadata/training_metadata.csv")
meta_test <-  read.csv("metadata/test_metadata.csv")

train_final <- left_join(meta_train, reduced_data, by = c("participant_id" = "participant_id"))
test_final  <- left_join(meta_test,  test.csv,  by = c("participant_id" = "participant_id"))

#####################################################################

## Make ML data frame where all categorical variables are treated as factors
ml_data <- train_final%>% 
  mutate(across(where(is.character), as.factor))

## split training data into training/test set and
set.seed(42)  # For reproducibility
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
ml_data_train <- ml_data[train_index, ]
ml_data_test <- ml_data[-train_index, ]

## Create set of predictors and response variable
X_train <- ml_data_train %>% select(-c(age, participant_id ))
y_train <- ml_data_train$age

##################################################################
##################################################################
## Fit a Random Forest Model

# Set seed for reproducibility
set.seed(42)

# 1. Create conservative recipe
rf_recipe <- recipe(age ~ ., data = ml_data_train) %>%
  update_role(participant_id, new_role = "ID") %>%
  step_nzv(all_predictors()) %>%
  step_corr(all_numeric(), threshold = 0.9) %>%
  step_normalize(all_numeric_predictors())

# 2. Define conservative model specification (REMOVED unsupported parameters)
rf_model <- rand_forest(
  mtry = tune(),
  trees = 500,
  min_n = tune()
) %>% 
  set_engine("ranger",
             importance = "permutation",
             splitrule = "variance",  # Changed from extratrees
             sample.fraction = 0.7,   # Subsample data per tree
             seed = 42) %>% 
  set_mode("regression")

# 3. Conservative tuning grid
tune_grid <- grid_regular(
  mtry(range = c(3, 10)),
  min_n(range = c(10, 30)),
  levels = 5
)

# 4. Create workflow
rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

# 5. Run conservative tuning
folds <- vfold_cv(ml_data_train, v = 5)
tuned_rf <- rf_workflow %>% 
  tune_grid(
    resamples = folds,
    grid = tune_grid,
    metrics = metric_set(rmse, mae, rsq)
  )

# 7. If no errors, proceed with model selection
best_rf <- tuned_rf %>% select_best(metric = "rmse")
final_rf <- rf_workflow %>% 
  finalize_workflow(best_rf) %>% 
  fit(ml_data_train)

## Now test model fit
train_preds <- predict(final_rf, ml_data_train) %>% 
  bind_cols(ml_data_train %>% select(age, participant_id))

# Calculate metrics
train_metrics <- train_preds %>% 
  metrics(truth = age, estimate = .pred)
print(train_metrics)

# 8. Evaluate on test set
test_preds <- predict(final_rf, ml_data_test) %>% 
  bind_cols(ml_data_test %>% select(age, participant_id))

test_metrics <- test_preds %>% 
  metrics(truth = age, estimate = .pred)

print(test_metrics)

# 9. Variable importance
vip(extract_fit_parsnip(final_rf), num_features = 20)

### Save model
saveRDS(final_rf, "regularized_rf_model.rds")

######################################################################
######################################################################
