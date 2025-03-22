//! Cluster detection algorithms

use anyhow::Result;
use crate::graph::CompressedGraph;
use crate::cluster::{Cluster, ClusterCentralNodes};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use log;
use rayon::prelude::*;

/// Thread-safe Union-Find data structure for efficient connected component analysis
pub struct DisjointSets {
    /// Parent pointers (parent[i] = parent of node i)
    parent: Vec<u32>,
    
    /// Rank/size of each set (for union by rank)
    rank: Vec<u32>,
}

impl DisjointSets {
    /// Create a new DisjointSets data structure
    pub fn new(size: usize) -> Self {
        let mut parent = Vec::with_capacity(size);
        let mut rank = Vec::with_capacity(size);
        
        // Initialize each node as its own set
        for i in 0..size {
            parent.push(i as u32);
            rank.push(1);
        }
        
        Self { parent, rank }
    }
    
    /// Find the root of the set containing x with path compression
    pub fn find(&mut self, x: u32) -> u32 {
        let px = self.parent[x as usize];
        if px != x {
            // Path compression: set parent to root
            self.parent[x as usize] = self.find(px);
        }
        self.parent[x as usize]
    }
    
    /// Union the sets containing x and y
    pub fn union(&mut self, x: u32, y: u32) {
        let root_x = self.find(x);
        let root_y = self.find(y);
        
        if root_x == root_y {
            return; // Already in the same set
        }
        
        // Union by rank: attach smaller tree under root of larger tree
        let rank_x = self.rank[root_x as usize];
        let rank_y = self.rank[root_y as usize];
        
        if rank_x > rank_y {
            self.parent[root_y as usize] = root_x;
            self.rank[root_x as usize] += self.rank[root_y as usize];
        } else {
            self.parent[root_x as usize] = root_y;
            self.rank[root_y as usize] += self.rank[root_x as usize];
        }
    }
    
    /// Get the size of the set containing x
    pub fn size(&mut self, x: u32) -> u32 {
        let root = self.find(x);
        self.rank[root as usize]
    }
}

/// Calculate cluster density (actual edges / potential edges) with parallel processing
/// for larger clusters
pub fn calculate_cluster_density(graph: &CompressedGraph, members: &[u32]) -> f32 {
    let n = members.len();
    if n <= 1 {
        return 1.0; // By convention, singleton clusters have density 1
    }
    
    // Potential edges = n * (n - 1) for directed graph
    let potential_edges = n * (n - 1);
    
    // For small clusters, use sequential processing
    if n < 1000 {
        return calculate_cluster_density_sequential(graph, members, potential_edges);
    }
    
    // For larger clusters, use parallel processing
    // Create a HashSet for efficient membership checking
    let members_set: std::collections::HashSet<u32> = members.iter().cloned().collect();
    
    // Count actual edges within the cluster in parallel
    let actual_edges: usize = members.par_iter()
        .map(|&src_idx| {
            let src = src_idx as usize;
            let mut local_edges = 0;
            
            for &dst_idx in graph.outgoing_edges(src) {
                // Check if destination is also in the cluster
                if members_set.contains(&dst_idx) {
                    local_edges += 1;
                }
            }
            
            local_edges
        })
        .sum();
    
    actual_edges as f32 / potential_edges as f32
}

/// Sequential version for smaller clusters
fn calculate_cluster_density_sequential(
    graph: &CompressedGraph, 
    members: &[u32],
    potential_edges: usize
) -> f32 {
    // Count actual edges within the cluster
    let mut actual_edges = 0;
    
    // Create a HashSet for more efficient lookups
    let members_set: std::collections::HashSet<u32> = members.iter().cloned().collect();
    
    for &src_idx in members {
        let src = src_idx as usize;
        for &dst_idx in graph.outgoing_edges(src) {
            // Check if destination is also in the cluster
            if members_set.contains(&dst_idx) {
                actual_edges += 1;
            }
        }
    }
    
    actual_edges as f32 / potential_edges as f32
}

/// Identify central nodes in a cluster using parallel processing
pub fn identify_central_nodes(graph: &CompressedGraph, members: &[u32]) -> ClusterCentralNodes {
    // For small clusters, use sequential processing
    if members.len() < 1000 {
        return identify_central_nodes_sequential(graph, members);
    }
    
    // For larger clusters, use parallel processing
    // Create a HashSet for quick lookups
    let members_set: std::collections::HashSet<u32> = members.iter().cloned().collect();
    
    // Calculate degrees in parallel
    let degrees: HashMap<u32, u32> = members.par_iter()
        .map(|&node_idx| {
            let node = node_idx as usize;
            // Start with out-degree to other members
            let mut degree = 0;
            
            for &dst in graph.outgoing_edges(node) {
                if members_set.contains(&dst) {
                    degree += 1;
                }
            }
            
            // Count in-degree from other members
            for &other_idx in members {
                if other_idx != node_idx && graph.has_edge(other_idx as usize, node_idx) {
                    degree += 1;
                }
            }
            
            (node_idx, degree)
        })
        .collect();
    
    // Find nodes with highest degree
    let mut nodes_by_degree: Vec<(u32, u32)> = degrees.into_iter().collect();
    nodes_by_degree.sort_unstable_by(|a, b| b.1.cmp(&a.1)); // Sort by degree (descending)
    
    // Take top 5 nodes (or all if less than 5)
    let top_n = std::cmp::min(5, nodes_by_degree.len());
    let degree_centrality = nodes_by_degree.iter()
        .take(top_n)
        .map(|&(node, _)| node)
        .collect();
    
    // For now, only compute degree centrality
    ClusterCentralNodes {
        degree: degree_centrality,
        betweenness: None,
        closeness: None,
    }
}

/// Sequential version for smaller clusters
fn identify_central_nodes_sequential(graph: &CompressedGraph, members: &[u32]) -> ClusterCentralNodes {
    // Count in-degree and out-degree for each node in the cluster
    let mut degrees = HashMap::new();
    
    for &node_idx in members {
        let node = node_idx as usize;
        // Start with out-degree
        let mut degree = 0;
        
        for &dst in graph.outgoing_edges(node) {
            if members.contains(&dst) {
                degree += 1;
            }
        }
        
        // Add in-degree (other nodes pointing to this one)
        for &other_idx in members {
            let other = other_idx as usize;
            if node_idx != other_idx && graph.has_edge(other, node_idx) {
                degree += 1;
            }
        }
        
        degrees.insert(node_idx, degree);
    }
    
    // Find nodes with highest degree
    let mut nodes_by_degree: Vec<(u32, u32)> = degrees.into_iter().collect();
    nodes_by_degree.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by degree (descending)
    
    // Take top 5 nodes (or all if less than 5)
    let top_n = std::cmp::min(5, nodes_by_degree.len());
    let degree_centrality = nodes_by_degree.iter()
        .take(top_n)
        .map(|&(node, _)| node)
        .collect();
    
    // For now, only compute degree centrality
    ClusterCentralNodes {
        degree: degree_centrality,
        betweenness: None,
        closeness: None,
    }
}

/// Find connected components in the graph using CPU with parallel processing
pub fn find_connected_components_cpu(
    graph: &CompressedGraph,
    min_cluster_size: usize,
) -> Result<Vec<Cluster>> {
    log::info!("Finding connected components using parallel CPU implementation");
    
    let node_count = graph.node_count;
    let disjoint_sets = Arc::new(Mutex::new(DisjointSets::new(node_count)));
    
    // Process nodes in parallel chunks for better load balancing
    let chunk_size = 10000; // Adjust based on graph size
    let num_chunks = (node_count + chunk_size - 1) / chunk_size;
    
    log::info!("Processing graph with {} nodes in {} chunks", node_count, num_chunks);
    
    // First phase: Process each chunk in parallel to find local connections
    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, node_count);
        
        let mut local_unions = Vec::new();
        
        // Collect all unions needed for this chunk
        for node in start..end {
            let src = node as u32;
            for &dst in graph.outgoing_edges(node) {
                local_unions.push((src, dst));
            }
        }
        
        // Apply unions in batches to reduce lock contention
        if !local_unions.is_empty() {
            let mut sets = disjoint_sets.lock().unwrap();
            for (src, dst) in local_unions {
                sets.union(src, dst);
            }
        }
    });
    
    // Group nodes by component
    log::info!("Grouping nodes by connected component");
    
    // Process in parallel chunks and combine results
    let component_maps: Vec<HashMap<u32, Vec<u32>>> = (0..num_chunks).into_par_iter().map(|chunk_idx| {
        let start = chunk_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, node_count);
        
        let mut local_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut sets = disjoint_sets.lock().unwrap();
        
        for node in start..end {
            let node_u32 = node as u32;
            let root = sets.find(node_u32);
            local_map.entry(root).or_default().push(node_u32);
        }
        
        local_map
    }).collect();
    
    // Combine all component maps
    let mut combined_map: HashMap<u32, Vec<u32>> = HashMap::new();
    for map in component_maps {
        for (root, mut nodes) in map {
            combined_map.entry(root).or_default().append(&mut nodes);
        }
    }
    
    // Filter by minimum size and create cluster objects in parallel
    log::info!("Creating {} clusters", combined_map.len());
    
    let clusters_vec: Vec<(u32, Vec<u32>)> = combined_map.into_iter()
        .filter(|(_, members)| members.len() >= min_cluster_size)
        .collect();
    
    let clusters: Vec<Cluster> = clusters_vec.into_par_iter()
        .enumerate()
        .map(|(id, (_, members))| {
            let density = calculate_cluster_density(graph, &members);
            let central_nodes = identify_central_nodes(graph, &members);
            Cluster {
                id: id as u32,
                size: members.len(),
                members,
                density,
                central_nodes,
            }
        })
        .collect();
    
    // Sort clusters by size (largest first)
    let mut sorted_clusters = clusters;
    sorted_clusters.sort_by(|a, b| b.size.cmp(&a.size));
    
    log::info!(
        "Found {} clusters with {} or more members",
        sorted_clusters.len(),
        min_cluster_size
    );
    
    Ok(sorted_clusters)
}

