//! Parquet file handling for graph data

use std::collections::{HashMap, HashSet};
use anyhow::Result;
use polars::prelude::*;
use crate::graph::CompressedGraph;
use log;

/// Load Farcaster links data with minimal memory usage
pub fn load_follow_data(
    path: &str,
    min_followings: usize,
    _sample_ratio: f32,
    _chunk_size: usize,
) -> Result<CompressedGraph> {
    // Load the parquet file
    log::info!("Reading parquet file: {}", path);
    
    // Check if the file exists
    if !std::path::Path::new(path).exists() {
        return Err(anyhow::anyhow!("File not found: {}", path));
    }
    
    // We'll keep this approach simple - load entire dataset, filter, and process
    let df = LazyFrame::scan_parquet(path, Default::default())?
        .filter(col("LinkType").eq(lit("follow")))
        .collect()?;
    
    // Print schema to debug
    log::info!("File schema: {:?}", df.schema());
    log::info!("Loaded {} follow relationships", df.height());
    
    // Process the results
    // Count users by their followings - using DataFrame operations
    let mut user_followings = HashMap::new();
    let fid_col = df.column("Fid")?.str()?;
    
    // Count followings per user
    for i in 0..df.height() {
        if let Some(user) = fid_col.get(i) {
            *user_followings.entry(user.to_string()).or_insert(0) += 1;
        }
    }
    
    // Filter for active users with sufficient followings
    let active_users: HashSet<String> = user_followings.iter()
        .filter(|(_, &count)| count >= min_followings as i64)
        .map(|(user, _)| user.clone())
        .collect();
    
    log::info!("Found {} users with {} or more followings", active_users.len(), min_followings);
    
    // Build string ID to index mapping
    let mut id_to_index: HashMap<String, u32> = HashMap::with_capacity(active_users.len() * 2);
    let mut node_ids: Vec<String> = Vec::with_capacity(active_users.len() * 2);
    
    // Process the data to construct the graph
    log::info!("Building compressed graph representation...");
    
    // First, determine the degree of each node to preallocate
    let mut temp_degrees: Vec<u32> = Vec::new();
    let mut next_id: u32 = 0;
    
    // Get all columns we need for processing
    let fid_col = df.column("Fid")?.str()?;
    let target_fid_col = df.column("TargetFid")?.str()?;
    
    let row_count = df.height();
    log::info!("Processing {} follow relationships", row_count);
    
    // First pass: count degrees and assign node IDs
    for i in 0..row_count {
        let src = fid_col.get(i).unwrap_or_default();
        let dst = target_fid_col.get(i).unwrap_or_default();
        
        // Skip if source is not in active users
        if !active_users.contains(src) {
            continue;
        }
        
        // Ensure both source and target have node IDs
        let src_idx = *id_to_index.entry(src.to_string()).or_insert_with(|| {
            let idx = next_id;
            next_id += 1;
            node_ids.push(src.to_string());
            temp_degrees.resize(next_id as usize, 0);
            idx
        });
        
        // Only count the target if we'll include it in our graph
        let _ = id_to_index.entry(dst.to_string()).or_insert_with(|| {
            let idx = next_id;
            next_id += 1;
            node_ids.push(dst.to_string());
            temp_degrees.resize(next_id as usize, 0);
            idx
        });
        
        // Increment degree for this node
        temp_degrees[src_idx as usize] += 1;
    }
    
    // Allocate graph structure using exact sizes
    let node_count = next_id as usize;
    let edge_count: usize = temp_degrees.iter().map(|&d| d as usize).sum();
    
    log::info!("Allocating graph with {} nodes and {} edges", node_count, edge_count);
    
    let mut graph = CompressedGraph::with_capacity(node_count, edge_count);
    
    // Set up offsets
    graph.offsets.push(0);
    let mut offset = 0;
    for &degree in &temp_degrees {
        offset += degree;
        graph.offsets.push(offset);
    }
    
    // Create temporary counters for current insertion positions
    let mut current_offsets = vec![0; node_count];
    
    // Resize edges array to final size
    graph.edges.resize(edge_count, 0);
    
    // Second pass: fill the edge array
    for i in 0..row_count {
        let src = fid_col.get(i).unwrap_or_default();
        let dst = target_fid_col.get(i).unwrap_or_default();
        
        // Skip if source is not in active users
        if !active_users.contains(src) {
            continue;
        }
        
        // Get indices (they should already exist)
        if let (Some(&src_idx), Some(&dst_idx)) = (id_to_index.get(src), id_to_index.get(dst)) {
            // Calculate position in edge array
            let pos = graph.offsets[src_idx as usize] as usize + current_offsets[src_idx as usize];
            graph.edges[pos] = dst_idx;
            current_offsets[src_idx as usize] += 1;
        }
    }
    
    // Set node metadata
    graph.node_count = node_count;
    graph.node_ids = Some(node_ids);
    
    // Sort edges for each node (optimizes later operations)
    graph.sort_adjacency_lists();
    
    Ok(graph)
}