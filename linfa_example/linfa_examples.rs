use linfa::prelude::*;
use syn_crabs::setup_logging;
use linfa_datasets::{iris, diabetes};
use polars::prelude::*;
use hugging_datasets::{load_csv_dataset, split_X_y, train_test_split};
use std::error::Error;
use std::result::Result;


/// # Main function to load datasets, split them, and apply Linfa classification and regression models.
/// ## Usage
/// This example demonstrates how to load the Iris dataset for classification and the Diabetes dataset
/// for regression using the `hugging_datasets` crate for data processing and Linfa for machine learning tasks.
fn main() -> Result<(), Box<dyn Error>> {
    syn_crabs::setup_logging()
    info!("--- Iris Classification ---");

    // Load the Iris dataset
    let (iris_X, iris_y) = match iris() {
        (X, y) => (X, y),
        _ => return Err("Error loading Iris dataset.".into()),
    };

    // Convert to DataFrame
    let iris_df = match DataFrame::from_ndarray(&iris_X) {
        Ok(df) => df,
        Err(e) => return Err(format!("Error converting Iris dataset to DataFrame: {}", e).into()),
    };

    // Convert the target to Series
    let iris_target = Series::new("target", iris_y);

    // Split into train/test sets
    let (iris_X_train, iris_X_test, iris_y_train, iris_y_test) =
        match train_test_split(&iris_df, &iris_target, 0.2) {
            Ok((X_train, X_test, y_train, y_test)) => (X_train, X_test, y_train, y_test),
            Err(e) => return Err(format!("Error splitting Iris dataset: {}", e).into()),
        };

    info!("Iris dataset - Train X: {:?}", iris_X_train.head(Some(5)));
    info!("Iris dataset - Train y: {:?}", iris_y_train.head(Some(5)));

    info!("\n--- Diabetes Regression ---");

    // Load the Diabetes dataset
    let (diabetes_X, diabetes_y) = match diabetes() {
        (X, y) => (X, y),
        _ => return Err("Error loading Diabetes dataset.".into()),
    };

    // Convert to DataFrame
    let diabetes_df = match DataFrame::from_ndarray(&diabetes_X) {
        Ok(df) => df,
        Err(e) => return Err(format!("Error converting Diabetes dataset to DataFrame: {}", e).into()),
    };

    // Convert the target to Series
    let diabetes_target = Series::new("target", diabetes_y);

    // Split into train/test sets
    let (diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test) =
        match train_test_split(&diabetes_df, &diabetes_target, 0.2) {
            Ok((X_train, X_test, y_train, y_test)) => (X_train, X_test, y_train, y_test),
            Err(e) => return Err(format!("Error splitting Diabetes dataset: {}", e).into()),
        };

    info!("Diabetes dataset - Train X: {:?}", diabetes_X_train.head(Some(5)));
    info!("Diabetes dataset - Train y: {:?}", diabetes_y_train.head(Some(5)));

    Ok(())
}