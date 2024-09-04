use clap::{Command, Arg};
use dataloader_rs::lib::{CSVSecurityDataset, JSONSecurityDataset, ParquetSecurityDataset};
use syn_crabs::setup_logging;
use polars::prelude::*;
use std::error::Error;
use log::*;

fn main() -> Result<(), Box<dyn Error>> {
    syn_crabs::setup_logging();

    let matches = Command::new("Hugging Datasets")
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

    let (format, input_path, target_column, test_ratio): (&str, &str, &str, f32) = if matches.is_present("format") && matches.is_present("input") {
        let format = matches.value_of("format").unwrap();
        let input_path = matches.value_of("input").unwrap();
        let target_column = matches.value_of("target_column").unwrap();
        let test_ratio: f32 = matches.value_of("test_ratio").unwrap().parse()?;
        (format, input_path, target_column, test_ratio)
    } else {
        let input_path = "data/security_dataset.csv";
        let target_column = "target";
        let format = "csv";
        let test_ratio: f32 = 0.2;
        (format, input_path, target_column, test_ratio)
    };

    log::info!("Format: {}", format);
    log::info!("Input path: {}", input_path);
    log::info!("Target column: {}", target_column);
    log::info!("Test ratio: {}", test_ratio);

    let df = match format {
        "csv" => CSVSecurityDataset::from_csv(input_path)?,
        "json" => JSONSecurityDataset::from_json(input_path)?,
        "parquet" => ParquetSecurityDataset::from_parquet(input_path)?,
        _ => unreachable!(),
    };

    let mut shuffled_df = df.clone();
    shuffled_df.shuffle();

    let (X, y) = split_X_y(&shuffled_df, target_column)?;

    let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, test_ratio)?;

    log::info!("X_train shape: {} rows, {} cols", X_train.height(), X_train.width());
    log::info!("y_train length: {}", y_train.len());
    log::info!("X_test shape: {} rows, {} cols", X_test.height(), X_test.width());
    log::info!("y_test length: {}", y_test.len());

    Ok(())
}

/// Splits the DataFrame into X (features) and y (target).
fn split_X_y(df: &DataFrame, target_column: &str) -> Result<(DataFrame, Series), Box<dyn Error>> {
    let y = df.column(target_column)?.clone();
    let X = df.drop(target_column)?;
    Ok((X, y))
}

/// Splits the dataset into training and testing sets.
pub fn train_test_split(
    X: &DataFrame,
    y: &Series,
    test_ratio: f32,
) -> Result<(DataFrame, DataFrame, Series, Series), Box<dyn Error>> {
    let n = X.height();
    let test_size = (n as f32 * test_ratio).round() as usize;

    let train_size = n - test_size;
    let X_train = X.slice(0, train_size);
    let X_test = X.slice(train_size, test_size);
    let y_train = y.slice(0, train_size);
    let y_test = y.slice(train_size, test_size);

    Ok((X_train, X_test, y_train, y_test))
}
