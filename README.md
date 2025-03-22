# Graph Cluster Analyzer

A high-performance Rust-based tool for analyzing social graph clusters in Farcaster network data.

## Overview

Graph Cluster Analyzer processes Farcaster social graph data stored in Parquet format to identify and analyze clusters of mutually connected users. It uses a memory-efficient compressed graph representation to handle large datasets and implements parallel processing for performance optimization.

## Features

- Load and process Farcaster follow relationships from Parquet files
- Filter users based on minimum follow relationships
- Identify clusters in the social graph
- Memory-efficient graph representation
- Support for sampling large datasets
- Detailed logging and analysis

## Requirements

- Rust 1.56 or later
- At least 16GB RAM (more recommended for large datasets)
- Parquet file with Farcaster social graph data

## Dataset

The Farcaster links dataset is available at:
https://huggingface.co/datasets/jc4p/farcaster-links

Download the parquet file from this repository before running the analyzer.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jc4p/graph-cluster-analyzer.git
cd graph-cluster-analyzer
```

2. Build the project in release mode:

```bash
cargo build --release
```

The compiled binary will be available at `./target/release/graph-cluster-analyzer`.

## Usage

Run the analyzer with the following command:

```bash
./target/release/graph-cluster-analyzer --input <PARQUET_FILE> --output-dir <OUTPUT_DIRECTORY> [OPTIONS]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input Parquet file (required) | - |
| `--output-dir` | Output directory for results | `cluster_results` |
| `--min-followings` | Minimum number of followings for a user to be included | 50 |
| `--min-cluster-size` | Minimum cluster size to be included in results | 3 |
| `--sample` | Sample ratio (0.0-1.0) for testing with smaller datasets | 1.0 |
| `--chunk-size` | Processing chunk size | 5000000 |
| `--threads` | Number of worker threads (0 = use all available cores) | 0 |
| `--verbose` | Enable verbose logging | false |

### Example

```bash
./target/release/graph-cluster-analyzer --input farcaster_links.parquet --output-dir cluster_results_test --min-followings 10 --min-cluster-size 3 --sample 0.000001 --verbose
```

This command:
- Processes the `farcaster_links.parquet` file
- Outputs results to the `cluster_results_test` directory
- Includes users with at least 10 followings
- Identifies clusters with at least 3 members
- Uses a small sample (0.000001 or 0.0001%) of the data for testing
- Enables verbose logging

## Project Structure

The project is organized as follows:

- `src/main.rs`: CLI entry point and main execution flow
- `src/data/parquet.rs`: Parquet file handling for graph data
- `src/graph/`: Graph representation and algorithms
- `src/cluster/`: Cluster detection and analysis

## Note on CUDA Implementation

The project was originally designed with CUDA GPU acceleration in mind (as described in the implementation plan), but the CPU-only version has been implemented first. The CUDA implementation may be added in future updates for enhanced performance on NVIDIA GPUs.

## Performance Considerations

- For large datasets, consider using sampling (`--sample`) for initial testing
- Adjust the minimum followings threshold (`--min-followings`) to focus on more active users
- Use `--chunk-size` to control memory usage during processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 