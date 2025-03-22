use anyhow::Result;
use clap::Parser;

mod config;
mod data;
mod graph;
mod cluster;
mod storage;
mod viz;

#[derive(Parser, Debug)]
#[clap(
    name = "graph-cluster-analyzer",
    about = "High-performance graph cluster analysis of Farcaster data"
)]
struct Cli {
    /// Path to input Parquet file
    #[clap(long)]
    input: String,
    
    /// Output directory for results
    #[clap(long, default_value = "cluster_results")]
    output_dir: String,
    
    /// Minimum number of followings for a user to be included
    #[clap(long, default_value = "50")]
    min_followings: usize,
    
    /// Minimum cluster size
    #[clap(long, default_value = "3")]
    min_cluster_size: usize,
    
    /// Sample ratio (0.0-1.0) for testing with smaller datasets
    #[clap(long, default_value = "1.0")]
    sample: f32,
    
    /// Processing chunk size
    #[clap(long, default_value = "5000000")]
    chunk_size: usize,
    
    
    /// Skip visualizations
    #[clap(long)]
    skip_viz: bool,
    
    /// Number of worker threads (0 = use all available cores)
    #[clap(long, default_value = "0")]
    threads: usize,
    
    /// Verbose logging
    #[clap(long, short)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args = Cli::parse();
    
    // Configure logging
    let log_level = if args.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_millis()
        .init();
    
    
    // Set number of threads
    let num_threads = if args.threads > 0 {
        args.threads
    } else {
        // If threads = 0, use all available cores
        num_cpus::get()
    };
    
    log::info!("Using {} worker threads", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;
    
    log::info!("Starting graph cluster analysis");
    log::info!("Input: {}", args.input);
    log::info!("Output: {}", args.output_dir);
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;
    
    // 1. Load data
    let graph = data::parquet::load_follow_data(
        &args.input,
        args.min_followings,
        args.sample,
        args.chunk_size
    )?;
    
    log::info!("Loaded graph with {} nodes and {} edges", 
              graph.node_count, graph.edges.len());
    
    // 2. Find mutual follows
    let mutual_graph = graph::algorithms::find_mutual_follows_cpu(&graph)?;
    
    log::info!("Found mutual follow graph with {} edges", mutual_graph.edges.len());
    
    // 3. Find clusters
    let clusters = cluster::detection::find_connected_components_cpu(
        &mutual_graph, 
        args.min_cluster_size
    )?;
    
    log::info!("Found {} clusters", clusters.len());
    
    // 4. Save results
    storage::save_results(&clusters, &graph, &mutual_graph, &args.output_dir)?;
    
    // 5. Generate visualizations if requested
    if !args.skip_viz {
        viz::generate_visualizations(&clusters, &mutual_graph, &args.output_dir)?;
    }
    
    log::info!("Analysis complete. Results saved to {}", args.output_dir);
    
    Ok(())
}