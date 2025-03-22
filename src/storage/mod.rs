//! Results persistence module

use anyhow::Result;
use crate::cluster::Cluster;
use crate::graph::CompressedGraph;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use serde_json::{json, to_string_pretty};

/// Save analysis results to the specified directory
pub fn save_results(
    clusters: &[Cluster],
    original_graph: &CompressedGraph,
    mutual_graph: &CompressedGraph,
    output_dir: &str,
) -> Result<()> {
    log::info!("Saving {} clusters to {}", clusters.len(), output_dir);
    
    // Ensure output directory exists
    fs::create_dir_all(output_dir)?;
    
    // Save summary information
    save_summary(clusters, original_graph, mutual_graph, output_dir)?;
    
    // Save each cluster
    save_clusters(clusters, original_graph, output_dir)?;
    
    // Save graph statistics
    save_graph_stats(original_graph, mutual_graph, output_dir)?;
    
    log::info!("Results saved successfully");
    
    Ok(())
}

/// Save summary information
fn save_summary(
    clusters: &[Cluster],
    original_graph: &CompressedGraph,
    mutual_graph: &CompressedGraph,
    output_dir: &str,
) -> Result<()> {
    log::info!("Saving summary information");
    
    let path = Path::new(output_dir).join("summary.json");
    let mut file = File::create(path)?;
    
    // Create summary object
    let summary = json!({
        "graph_stats": {
            "node_count": original_graph.node_count,
            "edge_count": original_graph.edges.len(),
            "mutual_edge_count": mutual_graph.edges.len(),
            "avg_degree": original_graph.edges.len() as f64 / original_graph.node_count as f64,
            "avg_mutual_degree": mutual_graph.edges.len() as f64 / mutual_graph.node_count as f64,
        },
        "cluster_stats": {
            "cluster_count": clusters.len(),
            "total_clustered_nodes": clusters.iter().map(|c| c.size).sum::<usize>(),
            "largest_cluster_size": clusters.get(0).map_or(0, |c| c.size),
            "smallest_cluster_size": clusters.last().map_or(0, |c| c.size),
            "avg_cluster_size": clusters.iter().map(|c| c.size).sum::<usize>() as f64 / 
                                if clusters.is_empty() { 1.0 } else { clusters.len() as f64 },
            "avg_density": clusters.iter().map(|c| c.density as f64).sum::<f64>() / 
                           if clusters.is_empty() { 1.0 } else { clusters.len() as f64 },
        }
    });
    
    file.write_all(to_string_pretty(&summary)?.as_bytes())?;
    
    Ok(())
}

/// Save individual cluster information
fn save_clusters(
    clusters: &[Cluster],
    original_graph: &CompressedGraph,
    output_dir: &str,
) -> Result<()> {
    log::info!("Saving individual cluster information");
    
    // Create clusters directory
    let clusters_dir = Path::new(output_dir).join("clusters");
    fs::create_dir_all(&clusters_dir)?;
    
    // Create a JSON file for each cluster
    for cluster in clusters {
        let path = clusters_dir.join(format!("cluster_{}.json", cluster.id));
        let mut file = File::create(path)?;
        
        // Resolve node IDs if available
        let member_ids = if let Some(ref node_ids) = original_graph.node_ids {
            cluster.members.iter()
                .map(|&id| node_ids[id as usize].clone())
                .collect::<Vec<_>>()
        } else {
            cluster.members.iter().map(|&id| id.to_string()).collect()
        };
        
        // Create cluster object
        let central_nodes_degree = if let Some(ref node_ids) = original_graph.node_ids {
            cluster.central_nodes.degree.iter()
                .map(|&id| node_ids[id as usize].clone())
                .collect::<Vec<_>>()
        } else {
            cluster.central_nodes.degree.iter().map(|&id| id.to_string()).collect()
        };
        
        let cluster_json = json!({
            "id": cluster.id,
            "size": cluster.size,
            "density": cluster.density,
            "central_nodes": {
                "degree": central_nodes_degree
            },
            "members": member_ids
        });
        
        file.write_all(to_string_pretty(&cluster_json)?.as_bytes())?;
    }
    
    // Create a JSON file with all clusters
    let all_clusters_path = Path::new(output_dir).join("all_clusters.json");
    let mut all_clusters_file = File::create(all_clusters_path)?;
    
    let clusters_json = json!({
        "clusters": clusters.iter().map(|c| {
            json!({
                "id": c.id,
                "size": c.size,
                "density": c.density,
                "central_nodes_count": c.central_nodes.degree.len()
            })
        }).collect::<Vec<_>>()
    });
    
    all_clusters_file.write_all(to_string_pretty(&clusters_json)?.as_bytes())?;
    
    Ok(())
}

/// Save graph statistics
fn save_graph_stats(
    original_graph: &CompressedGraph,
    mutual_graph: &CompressedGraph,
    output_dir: &str,
) -> Result<()> {
    log::info!("Saving graph statistics");
    
    let path = Path::new(output_dir).join("graph_stats.json");
    let mut file = File::create(path)?;
    
    // Calculate degree distribution
    let mut degree_dist = vec![0; 101]; // 0-100+ buckets
    let mut mutual_degree_dist = vec![0; 101];
    
    for node in 0..original_graph.node_count {
        let degree = original_graph.out_degree(node);
        let bucket = std::cmp::min(degree, 100);
        degree_dist[bucket] += 1;
        
        let mutual_degree = mutual_graph.out_degree(node);
        let mutual_bucket = std::cmp::min(mutual_degree, 100);
        mutual_degree_dist[mutual_bucket] += 1;
    }
    
    // Create graph stats object
    let stats = json!({
        "node_count": original_graph.node_count,
        "edge_count": original_graph.edges.len(),
        "mutual_edge_count": mutual_graph.edges.len(),
        "degree_distribution": degree_dist,
        "mutual_degree_distribution": mutual_degree_dist,
        "avg_degree": original_graph.edges.len() as f64 / original_graph.node_count as f64,
        "avg_mutual_degree": mutual_graph.edges.len() as f64 / mutual_graph.node_count as f64
    });
    
    file.write_all(to_string_pretty(&stats)?.as_bytes())?;
    
    Ok(())
}
