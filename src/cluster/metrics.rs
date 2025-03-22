//! Cluster statistics and metrics

use crate::graph::CompressedGraph;
use crate::cluster::Cluster;
use std::collections::HashMap;

/// Calculate various metrics for a cluster
pub fn calculate_cluster_metrics(
    cluster: &mut Cluster,
    graph: &CompressedGraph,
) {
    // Already calculated during cluster creation, but could be recalculated if needed
    cluster.density = calculate_density(graph, &cluster.members);
    
    // Calculate centrality measures
    calculate_centrality_measures(cluster, graph);
}

/// Calculate density (actual edges / potential edges)
pub fn calculate_density(
    graph: &CompressedGraph,
    members: &[u32],
) -> f32 {
    let n = members.len();
    if n <= 1 {
        return 1.0; // By convention, singleton clusters have density 1
    }
    
    // Potential edges = n * (n - 1) for directed graph
    let potential_edges = n * (n - 1);
    
    // Count actual edges within the cluster
    let mut actual_edges = 0;
    
    // Create a set of cluster members for quick lookup
    let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();
    
    for &src_idx in members {
        let src = src_idx as usize;
        for &dst_idx in graph.outgoing_edges(src) {
            // Check if destination is also in the cluster
            if member_set.contains(&dst_idx) {
                actual_edges += 1;
            }
        }
    }
    
    actual_edges as f32 / potential_edges as f32
}

/// Calculate centrality measures for a cluster
pub fn calculate_centrality_measures(
    cluster: &mut Cluster,
    graph: &CompressedGraph,
) {
    // Calculate degree centrality (in-degree + out-degree)
    let members = &cluster.members;
    
    // Create mapping for efficient lookups
    let mut idx_to_pos: HashMap<u32, usize> = HashMap::new();
    for (i, &node_idx) in members.iter().enumerate() {
        idx_to_pos.insert(node_idx, i);
    }
    
    // Calculate degree for each node in the cluster
    let mut degrees: Vec<(u32, u32)> = Vec::with_capacity(members.len());
    
    for &node_idx in members {
        let node = node_idx as usize;
        
        // Out-degree (links from this node to others in the cluster)
        let mut degree = 0;
        for &target in graph.outgoing_edges(node) {
            if idx_to_pos.contains_key(&target) {
                degree += 1;
            }
        }
        
        // In-degree (links to this node from others in the cluster)
        for &other_idx in members {
            let other = other_idx as usize;
            if node_idx != other_idx && graph.has_edge(other, node_idx) {
                degree += 1;
            }
        }
        
        degrees.push((node_idx, degree));
    }
    
    // Sort by degree (highest first)
    degrees.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Take top 5 (or fewer if cluster is smaller)
    let top_n = std::cmp::min(5, degrees.len());
    cluster.central_nodes.degree = degrees.iter()
        .take(top_n)
        .map(|&(node, _)| node)
        .collect();
    
    // For now, we're not calculating betweenness or closeness centrality
    // as they're computationally expensive. They could be implemented later.
    cluster.central_nodes.betweenness = None;
    cluster.central_nodes.closeness = None;
}