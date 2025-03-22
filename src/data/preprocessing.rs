//! Data preprocessing module for graph analysis

use anyhow::Result;
use crate::graph::CompressedGraph;

/// Extract a subgraph from a larger graph by node range
pub fn extract_subgraph(
    graph: &CompressedGraph,
    start_node: usize,
    end_node: usize,
) -> CompressedGraph {
    let subgraph_size = end_node - start_node;
    
    // Create mapping from original to subgraph indices
    let mut orig_to_sub = vec![u32::MAX; graph.node_count];
    for i in 0..subgraph_size {
        orig_to_sub[start_node + i] = i as u32;
    }
    
    // Count edges in the subgraph
    let mut edge_count = 0;
    for node in start_node..end_node {
        for &target in graph.outgoing_edges(node) {
            let target_idx = target as usize;
            // Only include edges where both endpoints are in the subgraph
            if target_idx >= start_node && target_idx < end_node {
                edge_count += 1;
            }
        }
    }
    
    // Create subgraph
    let mut subgraph = CompressedGraph::with_capacity(subgraph_size, edge_count);
    
    // Fill offsets and edges
    subgraph.offsets.push(0);
    let mut offset = 0;
    
    for node in start_node..end_node {
        for &target in graph.outgoing_edges(node) {
            let target_idx = target as usize;
            // Only include edges where both endpoints are in the subgraph
            if target_idx >= start_node && target_idx < end_node {
                subgraph.edges.push(orig_to_sub[target_idx]);
                offset += 1;
            }
        }
        subgraph.offsets.push(offset);
    }
    
    // Set node metadata
    subgraph.node_count = subgraph_size;
    
    // Copy node IDs if available
    if let Some(node_ids) = &graph.node_ids {
        let sub_node_ids = node_ids[start_node..end_node].to_vec();
        subgraph.node_ids = Some(sub_node_ids);
    }
    
    subgraph
}

/// Filter a graph to only include nodes with a minimum degree
pub fn filter_by_degree(
    graph: &CompressedGraph,
    min_degree: usize,
) -> Result<CompressedGraph> {
    // Count nodes that pass the filter
    let mut pass_filter = vec![false; graph.node_count];
    let mut filtered_count = 0;
    
    for node in 0..graph.node_count {
        if graph.out_degree(node) >= min_degree {
            pass_filter[node] = true;
            filtered_count += 1;
        }
    }
    
    // Create mapping from original to filtered indices
    let mut orig_to_filtered = vec![u32::MAX; graph.node_count];
    let mut filtered_idx = 0;
    for node in 0..graph.node_count {
        if pass_filter[node] {
            orig_to_filtered[node] = filtered_idx;
            filtered_idx += 1;
        }
    }
    
    // Count edges in the filtered graph
    let mut edge_count = 0;
    for node in 0..graph.node_count {
        if !pass_filter[node] {
            continue;
        }
        
        for &target in graph.outgoing_edges(node) {
            let target_idx = target as usize;
            // Only include edges where both endpoints pass the filter
            if pass_filter[target_idx] {
                edge_count += 1;
            }
        }
    }
    
    // Create filtered graph
    let mut filtered_graph = CompressedGraph::with_capacity(filtered_count, edge_count);
    
    // Fill offsets and edges
    filtered_graph.offsets.push(0);
    let mut offset = 0;
    
    for node in 0..graph.node_count {
        if !pass_filter[node] {
            continue;
        }
        
        for &target in graph.outgoing_edges(node) {
            let target_idx = target as usize;
            // Only include edges where both endpoints pass the filter
            if pass_filter[target_idx] {
                filtered_graph.edges.push(orig_to_filtered[target_idx]);
                offset += 1;
            }
        }
        filtered_graph.offsets.push(offset);
    }
    
    // Set node metadata
    filtered_graph.node_count = filtered_count;
    
    // Copy node IDs if available
    if let Some(node_ids) = &graph.node_ids {
        let mut filtered_node_ids = Vec::with_capacity(filtered_count);
        for node in 0..graph.node_count {
            if pass_filter[node] {
                filtered_node_ids.push(node_ids[node].clone());
            }
        }
        filtered_graph.node_ids = Some(filtered_node_ids);
    }
    
    Ok(filtered_graph)
}