//! Graph algorithms for analysis

use anyhow::Result;
use crate::graph::CompressedGraph;
use log;

/// Find mutual follow relationships (CPU implementation)
pub fn find_mutual_follows_cpu(graph: &CompressedGraph) -> Result<CompressedGraph> {
    log::info!("Finding mutual follows using CPU implementation");
    
    let node_count = graph.node_count;
    
    // Count mutual follows for each node
    let mut mutual_counts = vec![0u32; node_count];
    
    for src in 0..node_count {
        for &dst in graph.outgoing_edges(src) {
            let dst_idx = dst as usize;
            
            // Skip self-loops
            if dst_idx == src {
                continue;
            }
            
            // Check if dst also follows src
            if graph.has_edge(dst_idx, src as u32) {
                // Only count in one direction to avoid duplicates
                if src < dst_idx {
                    mutual_counts[src] += 1;
                }
            }
        }
    }
    
    // Calculate offsets
    let mut mutual_offsets = Vec::with_capacity(node_count + 1);
    mutual_offsets.push(0);
    
    let mut current_offset = 0;
    for &count in &mutual_counts {
        current_offset += count;
        mutual_offsets.push(current_offset);
    }
    
    // Allocate edges array
    let total_mutual_edges = current_offset as usize;
    log::info!("Found {} mutual follow relationships", total_mutual_edges);
    
    let mut mutual_edges = vec![0u32; total_mutual_edges];
    
    // Reset counters for filling array
    let mut current_pos = vec![0; node_count];
    
    // Fill in mutual edges
    for src in 0..node_count {
        for &dst in graph.outgoing_edges(src) {
            let dst_idx = dst as usize;
            
            // Skip self-loops
            if dst_idx == src {
                continue;
            }
            
            // Check if dst also follows src
            if graph.has_edge(dst_idx, src as u32) {
                // Only store in one direction
                if src < dst_idx {
                    let pos = mutual_offsets[src] as usize + current_pos[src];
                    mutual_edges[pos] = dst;
                    current_pos[src] += 1;
                }
            }
        }
    }
    
    // Create mutual graph
    let mutual_graph = CompressedGraph {
        node_count,
        offsets: mutual_offsets,
        edges: mutual_edges,
        node_ids: graph.node_ids.clone(),
        metadata: graph.metadata.clone(),
    };
    
    Ok(mutual_graph)
}


/// Optimize a graph for better cache locality by reordering nodes
pub fn optimize_for_cache(graph: &mut CompressedGraph) -> Result<()> {
    log::info!("Optimizing graph for cache locality");
    
    // Step 1: Reindex nodes for better locality
    let node_count = graph.node_count;
    
    // Build adjacency matrix density to guide reordering
    let mut node_degrees = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let degree = (graph.offsets[i+1] - graph.offsets[i]) as usize;
        node_degrees.push((i, degree));
    }
    
    // Sort nodes by degree for better cache locality
    node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Create new indexing
    let mut old_to_new = vec![0; node_count];
    for (new_idx, &(old_idx, _)) in node_degrees.iter().enumerate() {
        old_to_new[old_idx] = new_idx;
    }
    
    // Create a new graph with reordered nodes
    let mut new_graph = CompressedGraph::with_capacity(node_count, graph.edges.len());
    
    // Initialize offsets
    new_graph.offsets.push(0);
    let mut current_offset = 0;
    
    // Build new adjacency lists
    for new_idx in 0..node_count {
        let old_idx = node_degrees[new_idx].0;
        let start = graph.offsets[old_idx] as usize;
        let end = graph.offsets[old_idx + 1] as usize;
        
        // Reindex edges and store in new graph
        let mut new_edges: Vec<(u32, u32)> = Vec::with_capacity(end - start);
        
        for edge_idx in start..end {
            let target_old = graph.edges[edge_idx] as usize;
            let target_new = old_to_new[target_old];
            new_edges.push((target_new as u32, graph.edges[edge_idx]));
        }
        
        // Sort by new index for better cache locality
        new_edges.sort_by_key(|&(idx, _)| idx);
        
        // Add to new graph
        for (_, edge) in new_edges {
            new_graph.edges.push(edge);
            current_offset += 1;
        }
        
        new_graph.offsets.push(current_offset);
    }
    
    // Replace original graph
    *graph = new_graph;
    
    Ok(())
}