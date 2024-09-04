//! # DataSet and DataLoader in Rust
//!
//! This module provides functionality for loading, saving, and transforming datasets in various formats
//! like CSV, JSON, and Parquet. It includes capabilities for handling `DataFrame` operations and splitting 
//! the dataset into `X` (features) and `y` (target) for machine learning tasks.

use parquet::file::reader::SerializedFileReader;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::sync::Arc;
use std::error::Error;
use std::io::{BufReader, BufWriter, Write};
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
    pub fn new(data: DataFrame) -> Self {
        let uuid = Uuid::new_v4();
        let timestamp = Local::now().format("%d-%m-%y-%H").to_string();
        DataSet { data, uuid, timestamp }
    }

    /// Load data from a file in CSV, JSON, or Parquet format and convert it into a `DataFrame`.
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
                    Series::new("uuid", records.iter().map(|r| r.uuid.to_string()).collect::<Vec<String>>()),
                    Series::new("timestamp", records.iter().map(|r| r.timestamp.clone()).collect::<Vec<String>>()),
                    Series::new("source_ip", records.iter().map(|r| r.source_ip.clone()).collect::<Vec<String>>()),
                    Series::new("destination_ip", records.iter().map(|r| r.destination_ip.clone()).collect::<Vec<String>>()),
                    Series::new("action", records.iter().map(|r| r.action.clone()).collect::<Vec<String>>()),
                    Series::new("protocol", records.iter().map(|r| r.protocol.clone()).collect::<Vec<String>>()),
                ])?;
                df
            }
            "parquet" => {
                let df = LazyFrame::scan_parquet(path, Default::default())?.collect()?;
                df
            }
            _ => return Err("Unsupported file format".into()),
        };

        Ok(df)
    }

    /// Save the dataset as a Parquet file.
    pub fn save_as_parquet<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let path = file_path.as_ref();
        df.write_parquet(path, Default::default())?;
        Ok(())
    }

    /// Save the dataset as a JSON file.
    pub fn export_as_json<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let json_file = OpenOptions::new().write(true).create(true).open(file_path)?;
        let mut writer = BufWriter::new(json_file);

        let json: Vec<_> = df.iter().map(|s| s.to_string()).collect(); // Convert DataFrame to rows of strings
        let json_data = serde_json::to_string(&json)?; // Serialize to JSON string
        writer.write_all(json_data.as_bytes())?;
        Ok(())
    }

    /// Save the dataset as a CSV file.
    pub fn save_as_csv<P: AsRef<std::path::Path>>(df: &DataFrame, file_path: P) -> Result<(), Box<dyn Error>> {
        let file = OpenOptions::new().write(true).create(true).open(file_path)?;
        CsvWriter::new(file)
            .has_header(true)
            .finish(df)?;
        Ok(())
    }

    /// Save the dataset in the desired format (CSV, JSON, or Parquet).
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
