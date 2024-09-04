# Datasets Rustified 🔥🤗

A Rust implementation inspired by Hugging Face's `datasets` library, built for efficiency, scalability, and concurrent processing of large datasets. This library provides utilities to load data from multiple formats—CSV, JSON, and Parquet—and transform them into `DataFrame` objects using `Polars`. It also supports splitting datasets into training and test sets, offering flexibility for machine learning workflows. The library is designed to be highly concurrent, leveraging `Rayon` to maximize performance when handling large datasets.

### Features
- **Concurrent Data Loading**: Load datasets from CSV, JSON, and Parquet formats efficiently using `Polars` and `Rayon`.
- **Flexible Formats**: Supports input data from CSV, JSON, and Parquet formats, and converts them into a Polars `DataFrame` for further processing.
- **Train-Test Split**: Split datasets into training and testing sets with a user-specified test ratio.
- **UUID & Timestamp**: Each dataset session is uniquely identified with a UUID and timestamp, making dataset tracking and auditing seamless.

### Operations

- **Loading a Dataset**:
  - Load a dataset from a CSV file:
    ```rust
    let data_dict = DataDict::from_csv("dataset.csv")?;
    ```
  - Load a dataset from a JSON file:
    ```rust
    let data_dict = DataDict::from_json("dataset.json")?;
    ```
  - Load a dataset from a Parquet file:
    ```rust
    let data_dict = DataDict::from_parquet("dataset.parquet")?;
    ```

- **Splitting a Dataset**:
  - Split a dataset into train and test sets with a test ratio of 20%:
    ```rust
    let (train_set, test_set) = data_dict.train_test_split(0.2);
    ```

- **Saving a Dataset**:
  - Save the dataset as CSV:
    ```rust
    train_set.save_as_csv("train_set.csv")?;
    ```
  - Save the dataset as JSON:
    ```rust
    train_set.save_as_json("train_set.json")?;
    ```
  - Save the dataset as Parquet:
    ```rust
    train_set.save_as_parquet("train_set.parquet")?;
    ```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Contributing

We welcome contributions to `Datasets Rustified`! Feel free to submit issues, feature requests, or pull requests. When contributing, please ensure that you follow Rust's best practices and format code with `rustfmt`. Contributions that improve performance, add new data formats, or enhance documentation are greatly appreciated.

### Improvements
  - **Additional Formats**: Extend support to more formats like Arrow, HDF5, and more complex dataset types.
  - **Customizable DataFrame Operations**: Add the ability for users to apply custom transformations on the DataFrame before or after loading.
  - **Lazy Data Loading**: Implement lazy loading of large datasets for memory-efficient processing.
  - **Benchmarking & Optimization**: Introduce performance benchmarks for various dataset sizes to highlight concurrency and efficiency improvements.
  - **Advanced Dataset Transformations**: Add support for data augmentation, shuffling, and batching commonly used in machine learning workflows.

---
