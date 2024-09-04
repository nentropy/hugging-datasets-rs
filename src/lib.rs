use clap::{App, Arg};
use dataloader_rs::src::lib::{CSVSecurityDataset, JSONSecurityDataset, ParquetSecurityDataset};
use syn_crabs::setup_logging;
use polars::prelude::*;
use std::error::Error;
use log::*;

//! # Hugging Datasets Library
//! 
//! The `hugging_datasets` library provides tools for loading, splitting, and processing datasets 
//! from various formats such as CSV, JSON, and Parquet. It also allows you to prepare datasets 
//! for machine learning tasks, like classification and regression, by splitting features (`X`) 
//! and target (`y`) and creating train/test datasets.
//! 
//! ## Features
//! 
//! - Load datasets from different formats (`csv`, `json`, `parquet`).
//! - Shuffle and split datasets into train and test sets.
//! - Easily integrate with libraries like `linfa` for machine learning tasks.
//! 
//! ## Example Usage
//! 
//! ```rust
//! use hugging_datasets::{load_csv_dataset, split_X_y, train_test_split};
//! use polars::prelude::*;
//! 
//! let df = load_csv_dataset("data.csv").unwrap();
//! let (X, y) = split_X_y(&df, "target").unwrap();
//! let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2).unwrap();
//! ```
//! 
//! This example demonstrates how to load a CSV dataset, split it into features and target, and 
//! then split the dataset into training and testing sets, ready for model training.
fn main() -> Result<(), Box<dyn Error>> {
    syn_crabs::setup_logging();
   
    let matches = clap::App::new("Hugging Datasets")
        .version("1.0")
        .about("Loads and processes datasets from various formats.")
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .takes_value(true)
                .possible_values(&["csv", "json", "parquet"])
                .help("Specify the input dataset format (csv, json, parquet)"),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .takes_value(true)
                .help("Input file path"),
        )
        .arg(
            Arg::new("target_column")
                .short('t')
                .long("target")
                .takes_value(true)
                .default_value("target")
                .help("The column name to use as the target (y)"),
        )
        .arg(
            Arg::new("test_ratio")
                .short('r')
                .long("test_ratio")
                .takes_value(true)
                .default_value("0.2")
                .help("Ratio of the test set (default: 0.2)"),
        )
        .get_matches();

    // Conditional branching: Use args from the CLI or default values
    let (format, input_path, target_column, test_ratio) = if matches.is_present("format") && matches.is_present("input") {
        // Use CLI arguments if provided
        let format = matches.value_of("format").unwrap();
        let input_path = matches.value_of("input").unwrap();
        let target_column = matches.value_of("target_column").unwrap();
        let test_ratio: f32 = matches.value_of("test_ratio").unwrap().parse()?;
        (format, input_path, target_column, test_ratio)
    } else {
        // Use default values if no CLI arguments are provided
        let input_path = "data/security_dataset.csv"; 
        let target_column = "target";                 
        let format = "csv";                          
        let test_ratio: f32 = 0.2; 
        (format, input_path, target_column, test_ratio)
    };

    // Log the input parameters
    log::info!("Format: {}", format);
    log::info!("Input path: {}", input_path);
    log::info!("Target column: {}", target_column);
    log::info!("Test ratio: {}", test_ratio);

    // Load and process the dataset based on the specified format
    let df = match format {
        "csv" => CSVSecurityDataset::from_csv(input_path)?,
        "json" => JSONSecurityDataset::from_json(input_path)?,
        "parquet" => ParquetSecurityDataset::from_parquet(input_path)?,
        _ => unreachable!(),  // This case should never occur due to clap validation
    };

    // Shuffle the dataset
    let mut shuffled_df = df.clone();
    shuffled_df.shuffle();

    // Split the dataset into X (features) and y (target)
    let (X, y) = split_X_y(&shuffled_df, target_column)?;

    // Split into train/test sets
    let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, test_ratio)?;

    // Output ready for further processing or linfa training
    log::info!("X_train: {:?}", X_train);
    log::info!("y_train: {:?}", y_train);
    log::info!("X_test: {:?}", X_test);
    log::info!("y_test: {:?}", y_test);

    Ok(())
}

/// Splits the DataFrame into X (features) and y (target).
/// X is a Dataframe, Y is a series
fn split_X_y<Series, DataFrame>(df: &DataFrame, target_column: &str) -> Result<(DataFrame, Series), Box<dyn Error>> {
    let y = df.column(target_column)?.clone();
    let X = df.drop(target_column)?;
    Ok((X, y))
}

/// Splits the dataset into training and testing sets.
///
/// # Arguments
///
///  `X` - The features DataFrame.
///  `y` - The target Series.
///  `test_ratio` - The ratio of the test set.
///
/// # Returns
///
/// A tuple containing the training and test sets for both X and y.
pub fn train_test_split<DataFrame, Series, F32>(
    X: &DataFrame,
    y: &Series,
    test_ratio: F32
) ->  Result<(DataFrame, DataFrame, Series, Series), Box<dyn Error>> {
    let n = X.height();
    let test_size = (n as f32 * test_ratio).round() as usize;

    // Split indices
    let train_size = n - test_size;
    let X_train = X.slice(0, train_size);
    let X_test = X.slice(train_size, test_size);
    let y_train = y.slice(0, train_size);
    let y_test = y.slice(train_size, test_size);

    Ok((X_train, X_test, y_train, y_test))
}
