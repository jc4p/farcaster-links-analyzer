//! Graph construction module

use anyhow::Result;
use crate::graph::CompressedGraph;
use crate::graph::compressed::NodeMetadata;
use std::collections::HashMap;

/// Builder for incrementally constructing a CompressedGraph
pub struct GraphBuilder {
    /// Number of nodes
    node_count: usize,
    
    /// Mapping from string IDs to node indices
    id_to_index: HashMap<String, u32>,
    
    /// Node string IDs
    node_ids: Vec<String>,
    
    /// Adjacency lists for each node
    adjacency_lists: Vec<Vec<u32>>,
    
    /// Follower counts
    follower_counts: Vec<u32>,
    
    /// Following counts
    following_counts: Vec<u32>,
}

impl GraphBuilder {
    /// Create a new graph builder with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            node_count: 0,
            id_to_index: HashMap::with_capacity(capacity),
            node_ids: Vec::with_capacity(capacity),
            adjacency_lists: Vec::with_capacity(capacity),
            follower_counts: Vec::with_capacity(capacity),
            following_counts: Vec::with_capacity(capacity),
        }
    }
    
    /// Get or create a node ID for the given string ID
    pub fn get_or_create_node(&mut self, id: &str) -> u32 {
        if let Some(&idx) = self.id_to_index.get(id) {
            return idx;
        }
        
        // Create a new node
        let idx = self.node_count as u32;
        self.id_to_index.insert(id.to_string(), idx);
        self.node_ids.push(id.to_string());
        self.adjacency_lists.push(Vec::new());
        self.follower_counts.push(0);
        self.following_counts.push(0);
        self.node_count += 1;
        
        idx
    }
    
    /// Add an edge from one node to another
    pub fn add_edge(&mut self, src_id: &str, dst_id: &str) {
        let src_idx = self.get_or_create_node(src_id);
        let dst_idx = self.get_or_create_node(dst_id);
        
        // Add edge
        self.adjacency_lists[src_idx as usize].push(dst_idx);
        
        // Update counts
        self.following_counts[src_idx as usize] += 1;
        self.follower_counts[dst_idx as usize] += 1;
    }
    
    /// Build the compressed graph
    pub fn build(mut self) -> Result<CompressedGraph> {
        // Count total edges
        let edge_count: usize = self.adjacency_lists.iter()
            .map(|list| list.len())
            .sum();
        
        // Create offsets array
        let mut offsets = Vec::with_capacity(self.node_count + 1);
        offsets.push(0);
        
        let mut offset = 0;
        for list in &self.adjacency_lists {
            offset += list.len() as u32;
            offsets.push(offset);
        }
        
        // Create edges array
        let mut edges = Vec::with_capacity(edge_count);
        for list in &mut self.adjacency_lists {
            // Sort for binary search efficiency
            list.sort_unstable();
            edges.extend_from_slice(list);
        }
        
        // Create node metadata
        let metadata = NodeMetadata {
            follower_counts: self.follower_counts,
            following_counts: self.following_counts,
        };
        
        // Create the compressed graph
        let graph = CompressedGraph {
            node_count: self.node_count,
            offsets,
            edges,
            node_ids: Some(self.node_ids),
            metadata: Some(metadata),
        };
        
        Ok(graph)
    }
}