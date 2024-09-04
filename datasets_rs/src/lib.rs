//! # Datasets Rustified: Core Functionality
//! 
//! This module provides the core functionality for loading, saving, and converting datasets
//! in various formats, including CSV, JSON, and Parquet. 
//! 
//! ## Features
//! 
//! - **Concurrent DataFrame Handling**: Uses `Polars` for efficient DataFrame processing.
//! - **Multiple Format Support**: Supports `.csv`, `.json`, and `.parquet` formats.
//! 
//! ## Usage
//!
//! ### Loading and Saving
//! The core module provides functions for loading datasets from CSV, JSON, and Parquet formats,
//! and saving them into one of these formats based on user input.

use std::error::Error;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use parquet::file::writer::SerializedFileWriter; // Correct Parquet writer import
use polars::prelude::*; // Importing Polars for DataFrame handling
use serde_json::to_string; // Serde for JSON serialization
use serde::{Serialize, Deserialize};

/// Saves a DataFrame as a Parquet file.
///
/// # Arguments
///
///  `df` - A reference to the DataFrame to be saved.
///  `file_path` - The path where the Parquet file will be written.
///
/// # Errors
///
/// Returns an `Err` if the file cannot be created or if the Parquet writing fails.
///
/// # Example
/// 
/// ```rust
/// let df = CsvReader::from_path("data.csv")?.infer_schema(None).finish()?;
/// save_as_parquet(&df, "output.parquet")?;
/// ```
pub fn save_as_parquet<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
    let path = file_path.as_ref();
    let file = OpenOptions::new().write(true).create(true).open(path)?;
    let writer = BufWriter::new(file);

    // Parquet writer implementation, you may need to use Polars' methods to write the dataframe in parquet
    let mut writer = ParquetWriter::new(writer)?;
    writer.write(&df)?;

    Ok(())
}

/// Exports a DataFrame as a JSON file.
///
/// # Arguments
///
///  `df` - A reference to the DataFrame to be exported.
///  `file_path` - The path where the JSON file will be written.
///
/// # Errors
///
/// Returns an `Err` if the file cannot be created or if the JSON serialization fails.
///
/// # Example
/// 
/// ```rust
/// let df = CsvReader::from_path("data.csv")?.infer_schema(None).finish()?;
/// export_as_json(&df, "output.json")?;
/// ```
pub fn export_as_json<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
    let json_file = OpenOptions::new().write(true).create(true).open(file_path)?;
    let mut file = BufWriter::new(json_file);

    // Converting DataFrame to JSON compatible structure
    let json_data = df.to_json()?;
    file.write_all(json_data.as_bytes())?;
    
    Ok(())
}

/// Saves a DataFrame as a CSV file.
///
/// # Arguments
///
///  `df` - A reference to the DataFrame to be saved.
///  `file_path` - The path where the CSV file will be written.
///
/// # Errors
///
/// Returns an `Err` if the file cannot be created or if the CSV writing fails.
///
/// # Example
/// 
/// ```rust
/// let df = CsvReader::from_path("data.csv")?.infer_schema(None).finish()?;
/// save_as_csv(&df, "output.csv")?;
/// ```
pub fn save_as_csv<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().write(true).create(true).open(file_path)?;
    
    // Writing CSV to file
    CsvWriter::new(file)
        .has_header(true)
        .finish(df)?;
        
    Ok(())
}

/// Loads a dataset from a CSV file and returns a DataFrame.
///
/// # Example
///
/// ```rust
/// let df = load_dataset("input.csv")?;
/// ```
pub fn load_dataset<P: AsRef<std::path::Path>>(file_path: P) -> Result<DataFrame, Box<dyn Error>> {
    let df = CsvReader::from_path(file_path)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;
    Ok(df)
}
