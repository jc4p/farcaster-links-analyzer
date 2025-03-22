//! Configuration management for the graph cluster analyzer

/// Default configuration for the graph cluster analyzer
pub struct Config {
    /// Minimum number of followings for a user to be included
    pub min_followings: usize,
    
    /// Minimum cluster size
    pub min_cluster_size: usize,
    
    /// Sample ratio for testing with smaller datasets
    pub sample_ratio: f32,
    
    /// Processing chunk size
    pub chunk_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_followings: 50,
            min_cluster_size: 3,
            sample_ratio: 1.0,
            chunk_size: 5_000_000,
        }
    }
}

impl Config {
    /// Create a new configuration with custom values
    pub fn new(
        min_followings: usize,
        min_cluster_size: usize,
        sample_ratio: f32,
        chunk_size: usize,
    ) -> Self {
        Self {
            min_followings,
            min_cluster_size,
            sample_ratio,
            chunk_size,
        }
    }
}
