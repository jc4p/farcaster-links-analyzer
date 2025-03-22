//! Cluster analysis module

pub mod detection;
pub mod metrics;

use serde::{Serialize, Deserialize};

/// Represents a cluster (connected component) in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Unique identifier for this cluster
    pub id: u32,
    
    /// Members of this cluster (node indices)
    pub members: Vec<u32>,
    
    /// Size of the cluster
    pub size: usize,
    
    /// Density: actual edges / potential edges
    pub density: f32,
    
    /// Central nodes using various centrality measures
    pub central_nodes: ClusterCentralNodes,
}

/// Key nodes in a cluster identified by different centrality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCentralNodes {
    /// Nodes with highest degree centrality
    pub degree: Vec<u32>,
    
    /// Nodes with highest betweenness centrality (if computed)
    pub betweenness: Option<Vec<u32>>,
    
    /// Nodes with highest closeness centrality (if computed)
    pub closeness: Option<Vec<u32>>,
}
