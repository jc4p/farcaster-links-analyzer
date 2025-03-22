//! Memory-efficient graph representation

use std::mem;
use serde::{Serialize, Deserialize};

/// Store node metadata separately to improve cache locality during traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Number of followers per node
    pub follower_counts: Vec<u32>,
    
    /// Number of following per node
    pub following_counts: Vec<u32>,
}

impl NodeMetadata {
    /// Calculate the memory usage of the metadata
    pub fn memory_usage(&self) -> usize {
        let follower_counts = self.follower_counts.capacity() * mem::size_of::<u32>();
        let following_counts = self.following_counts.capacity() * mem::size_of::<u32>();
        
        follower_counts + following_counts
    }
}

/// Compressed sparse representation of a directed graph optimized for memory efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGraph {
    /// Number of nodes in the graph
    pub node_count: usize,
    
    /// Offset array: index where each node's edges begin
    /// offsets[i] to offsets[i+1] defines the edge range for node i
    pub offsets: Vec<u32>,
    
    /// Edge array: concatenated lists of target nodes
    pub edges: Vec<u32>,
    
    /// Optional mapping from internal node IDs to original string IDs
    pub node_ids: Option<Vec<String>>,
    
    /// Optional node metadata (stored separately for cache efficiency)
    pub metadata: Option<NodeMetadata>,
}

impl CompressedGraph {
    /// Create a new graph with pre-allocated capacity
    pub fn with_capacity(node_count: usize, edge_count: usize) -> Self {
        Self {
            node_count,
            offsets: Vec::with_capacity(node_count + 1),
            edges: Vec::with_capacity(edge_count),
            node_ids: None,
            metadata: None,
        }
    }
    
    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, node: usize) -> &[u32] {
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        &self.edges[start..end]
    }
    
    /// Sort all adjacency lists (improves binary search performance)
    pub fn sort_adjacency_lists(&mut self) {
        for node in 0..self.node_count {
            let start = self.offsets[node] as usize;
            let end = self.offsets[node + 1] as usize;
            if start < end {
                self.edges[start..end].sort_unstable();
            }
        }
    }
    
    /// Check if there's an edge from src to dst
    pub fn has_edge(&self, src: usize, dst: u32) -> bool {
        let edges = self.outgoing_edges(src);
        edges.binary_search(&dst).is_ok()
    }
    
    /// Get out-degree of a node
    pub fn out_degree(&self, node: usize) -> usize {
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        end - start
    }
    
    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let base = mem::size_of::<Self>();
        let offsets = self.offsets.capacity() * mem::size_of::<u32>();
        let edges = self.edges.capacity() * mem::size_of::<u32>();
        
        // Add metadata if present
        let ids = self.node_ids.as_ref()
            .map(|ids| ids.iter().map(|s| s.capacity()).sum::<usize>())
            .unwrap_or(0);
            
        let metadata = self.metadata.as_ref()
            .map(|m| m.memory_usage())
            .unwrap_or(0);
            
        base + offsets + edges + ids + metadata
    }
}