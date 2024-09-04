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
mod load_dataset;

use std::error::Error;
use std::io::Write;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::BufWriter;
use parquet::file::writer::{FileWriter, SerializedFileWriter};
use polars::prelude::*;
use serde_json::to_string;

/// Defaults:


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
    ParquetWriter::new(writer).finish(df)?;
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
#[derive(Serialize, Deserialize)]
pub fn export_as_json<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
    let json_file = OpenOptions::new().write(true).create(true).open(file_path)?;
    let json_data = to_string(df)?;
    std::fs::write(json_file, json_data)?;
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
    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf).finish(&df)?;
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