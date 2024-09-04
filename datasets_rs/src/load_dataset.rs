//! # DataSet and DataLoader in Rust
//!
//! This module provides functionality for loading, saving, and transforming datasets in various formats
//! like CSV, JSON, and Parquet. It includes capabilities for handling `DataFrame` operations and splitting 
//! the dataset into `X` (features) and `y` (target) for machine learning tasks.
//!
//! ## Features
//!
//! - Load data from CSV, JSON, and Parquet formats.
//! - Save data in CSV, JSON, or Parquet formats.
//! - Identify a target feature for machine learning and split `X` (features) and `y` (target).
//!
//! ## Example Usage
//!
//! ### 1. Loading a Dataset
//! ```rust
//! let df = DataSet::load_data("dataset.csv").unwrap();
//! let dataset = DataSet::new(df);
//! ```
//!
//! ### 2. Saving a Dataset
//! ```rust
//! dataset.save_data("output.csv", "csv").unwrap();
//! ```

use arrow::record_batch::RecordBatch;
use parquet::arrow::{arrow_reader, ParquetFileArrowReader};
use parquet::file::writer::{FileWriter, SerializedFileWriter};
use parquet::schema::types::Type;
use serde::{Deserialize, Serialize};
use polars::prelude::*;
use polars::prelude::Series;
use std::fs::File;
use std::sync::Arc;
use std::path::PathBuf;
use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::fs::OpenOptions;
use uuid::Uuid;
use chrono::Local;

/// A structure that represents a single record in a security dataset.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SecurityRecord {
    pub uuid: Uuid,
    pub timestamp: String,
    pub source_ip: String,
    pub destination_ip: String,
    pub action: String,
    pub protocol: String,
}

/// The main `DataSet` structure used for storing and managing the dataset in a DataFrame.
pub struct DataSet {
    pub data: DataFrame,
    pub uuid: Uuid,
    pub timestamp: String,
}

impl DataSet {
    /// Create a new `DataSet` instance.
    ///
    /// # Arguments
    ///
    /// `data` - The data in the form of a `DataFrame`.
    ///
    /// # Returns
    ///
    /// A `DataSet` struct that contains the provided DataFrame and metadata like UUID and timestamp.
    pub fn new(data: DataFrame) -> Self {
        let uuid = Uuid::new_v4();
        let timestamp = Local::now().format("%d-%m-%y-%H").to_string();
        DataSet { data, uuid, timestamp }
    }

    /// Load data from a file in CSV, JSON, or Parquet format and convert it into a `DataFrame`.
    ///
    /// # Arguments
    ///
    /// `file_path` - Path to the file (CSV, JSON, or Parquet).
    ///
    /// # Returns
    ///
    /// A `Result` containing a `DataFrame` or an error.
    ///
    /// # Example
    ///
    /// ```rust
    /// let df = DataSet::load_data("data.csv").unwrap();
    /// ```
    pub fn load_data<P: AsRef<std::path::Path>>(file_path: P) -> Result<DataFrame, Box<dyn Error>> {
        let path = file_path.as_ref();
        let ext = path.extension().and_then(std::ffi::OsStr::to_str).unwrap_or("");

        let df = match ext {
            "csv" => CsvReader::from_path(path)?.infer_schema(None).finish()?,
            "json" => {
                let json_file = File::open(path)?;
                let reader = BufReader::new(json_file);
                let records: Vec<SecurityRecord> = serde_json::from_reader(reader)?;
                let df = DataFrame::new(vec![
                    Series::new("uuid", records.iter().map(|r| &r.uuid).collect::<Vec<&Uuid>>()),
                    Series::new("timestamp", records.iter().map(|r| &r.timestamp).collect::<Vec<&String>>()),
                    Series::new("source_ip", records.iter().map(|r| &r.source_ip).collect::<Vec<&String>>()),
                    Series::new("destination_ip", records.iter().map(|r| &r.destination_ip).collect::<Vec<&String>>()),
                    Series::new("action", records.iter().map(|r| &r.action).collect::<Vec<&String>>()),
                    Series::new("protocol", records.iter().map(|r| &r.protocol).collect::<Vec<&String>>()),
                ])?;
                df
            }
            "parquet" => {
                let parquet_file = File::open(path)?;
                let file_reader = SerializedFileReader::new(parquet_file)?;
                let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));
                let record_batch_reader = arrow_reader.get_record_reader(1024)?;
                let mut batches: Vec<RecordBatch> = Vec::new();
                for batch in record_batch_reader {
                    batches.push(batch?);
                }
                let df: DataFrame = DataFrame::from_parquet(&batches)?;
                Ok(df)
            }
            _ => return Err("Unsupported file format".into()),
        };

        Ok(df)
    }

    /// Save the dataset as a Parquet file.
    ///
    /// # Arguments
    ///
    ///  `df` - The `DataFrame` to save.
    ///  `file_path` - The path where the Parquet file will be saved.
    ///
    /// # Returns
    ///
    /// A `Result` that signifies success or contains an error.
    pub fn save_as_parquet<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let path = file_path.as_ref();
        let file = OpenOptions::new().write(true).create(true).open(path)?;
        let writer = BufWriter::new(file);
        ParquetWriter::new(writer).finish(df)?;
        Ok(())
    }

    /// Save the dataset as a JSON file.
    ///
    /// # Arguments
    ///
    ///  `df` - The `DataFrame` to save.
    ///  `file_path` - The path where the JSON file will be saved.
    ///
    /// # Returns
    ///
    /// A `Result` that signifies success or contains an error.
    pub fn export_as_json<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let json_file = File::create(file_path)?;
        let mut writer = BufWriter::new(json_file);
        let json = serde_json::to_string(&df)?;
        writer.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Save the dataset as a CSV file.
    ///
    /// # Arguments
    ///
    /// * `df` - The `DataFrame` to save.
    /// * `file_path` - The path where the CSV file will be saved.
    ///
    /// # Returns
    ///
    /// A `Result` that signifies success or contains an error.
    pub fn save_as_csv<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let file = OpenOptions::new().write(true).create(true).open(file_path)?;
        CsvWriter::new(file)
            .has_header(true)
            .finish(df)?;
        Ok(())
    }

    /// Save the dataset in the desired format (CSV, JSON, or Parquet).
    ///
    /// # Arguments
    ///
    ///  `file_path` - The path where the file will be saved.
    ///  `file_extension` - The format of the file (`csv`, `json`, or `parquet`).
    ///
    /// # Returns
    ///
    /// A `Result` that signifies success or contains an error.
    pub fn save_data<P: AsRef<std::path::Path>>(
        &self,
        file_path: P,
        file_extension: &str,
    ) -> Result<(), Box<dyn Error>> {
        match file_extension {
            "csv" => self::save_as_csv(&self.data, file_path)?,
            "json" => self::export_as_json(&self.data, file_path)?,
            "parquet" => self::save_as_parquet(&self.data, file_path)?,
            _ => return Err("Unsupported file format".into()),
        }
        Ok(())
    }
}
