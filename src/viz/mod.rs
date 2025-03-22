//! Visualization generation module

use anyhow::Result;
use crate::cluster::Cluster;
use crate::graph::CompressedGraph;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Generate visualizations from analysis results
pub fn generate_visualizations(
    clusters: &[Cluster],
    mutual_graph: &CompressedGraph,
    output_dir: &str,
) -> Result<()> {
    log::info!("Generating visualizations for {} clusters", clusters.len());
    
    // Create visualizations directory
    let viz_dir = Path::new(output_dir).join("visualizations");
    fs::create_dir_all(&viz_dir)?;
    
    // Generate network data files for visualization
    generate_network_data(clusters, mutual_graph, &viz_dir)?;
    
    // Generate HTML files for interactive visualization
    generate_html_visualizations(clusters, &viz_dir)?;
    
    // Generate statistics visualizations
    generate_stats_visualizations(clusters, &viz_dir)?;
    
    log::info!("Visualizations generated successfully");
    
    Ok(())
}

/// Generate network data files for visualization tools
fn generate_network_data(
    clusters: &[Cluster], 
    mutual_graph: &CompressedGraph,
    viz_dir: &Path
) -> Result<()> {
    log::info!("Generating network data files");
    
    // Create data directory
    let data_dir = viz_dir.join("data");
    fs::create_dir_all(&data_dir)?;
    
    // For each large cluster, create a GraphML file
    for cluster in clusters.iter().take(10) { // Only top 10 clusters
        if cluster.size < 10 {
            continue; // Skip small clusters
        }
        
        let file_path = data_dir.join(format!("cluster_{}_network.graphml", cluster.id));
        let mut file = File::create(file_path)?;
        
        // Write GraphML header
        writeln!(file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(file, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">")?;
        writeln!(file, "  <graph id=\"G\" edgedefault=\"directed\">")?;
        
        // Write nodes
        for &node_id in &cluster.members {
            let node_label = if let Some(ref node_ids) = mutual_graph.node_ids {
                &node_ids[node_id as usize]
            } else {
                "Unknown"
            };
            
            writeln!(file, "    <node id=\"n{}\">\n      <data key=\"label\">{}</data>\n    </node>", 
                     node_id, node_label)?;
        }
        
        // Write edges
        let mut edge_id = 0;
        for &src in &cluster.members {
            let src_idx = src as usize;
            
            for &dst in mutual_graph.outgoing_edges(src_idx) {
                // Only include edges within the cluster
                if cluster.members.contains(&dst) {
                    writeln!(file, "    <edge id=\"e{}\" source=\"n{}\" target=\"n{}\"/>", 
                             edge_id, src, dst)?;
                    edge_id += 1;
                }
            }
        }
        
        // Write GraphML footer
        writeln!(file, "  </graph>")?;
        writeln!(file, "</graphml>")?;
    }
    
    // Create a CSV file with node data
    let nodes_file_path = data_dir.join("nodes.csv");
    let mut nodes_file = File::create(nodes_file_path)?;
    
    // Write header
    writeln!(nodes_file, "id,label,cluster_id")?;
    
    // Write nodes with cluster assignments
    for cluster in clusters {
        for &node_id in &cluster.members {
            let node_label = if let Some(ref node_ids) = mutual_graph.node_ids {
                &node_ids[node_id as usize]
            } else {
                "Unknown"
            };
            
            writeln!(nodes_file, "{},{},{}", node_id, node_label, cluster.id)?;
        }
    }
    
    Ok(())
}

/// Generate HTML files for interactive visualization
fn generate_html_visualizations(
    clusters: &[Cluster], 
    viz_dir: &Path
) -> Result<()> {
    log::info!("Generating HTML visualizations");
    
    // Create HTML directory
    let html_dir = viz_dir.join("html");
    fs::create_dir_all(&html_dir)?;
    
    // Create an index.html file
    let index_path = html_dir.join("index.html");
    let mut index_file = File::create(index_path)?;
    
    // Write a basic HTML file with cluster information
    writeln!(index_file, "<!DOCTYPE html>")?;
    writeln!(index_file, "<html lang=\"en\">")?;
    writeln!(index_file, "<head>")?;
    writeln!(index_file, "  <meta charset=\"UTF-8\">")?;
    writeln!(index_file, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")?;
    writeln!(index_file, "  <title>Farcaster Cluster Analysis</title>")?;
    writeln!(index_file, "  <style>")?;
    writeln!(index_file, "    body {{ font-family: Arial, sans-serif; margin: 20px; }}")?;
    writeln!(index_file, "    h1, h2 {{ color: #333; }}")?;
    writeln!(index_file, "    .cluster-list {{ display: flex; flex-wrap: wrap; }}")?;
    writeln!(index_file, "    .cluster-card {{ border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; width: 300px; }}")?;
    writeln!(index_file, "    .cluster-card h3 {{ margin-top: 0; }}")?;
    writeln!(index_file, "    .stats {{ margin-top: 20px; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}")?;
    writeln!(index_file, "  </style>")?;
    writeln!(index_file, "</head>")?;
    writeln!(index_file, "<body>")?;
    writeln!(index_file, "  <h1>Farcaster Cluster Analysis</h1>")?;
    
    // Write cluster statistics
    writeln!(index_file, "  <div class=\"stats\">")?;
    writeln!(index_file, "    <h2>Summary Statistics</h2>")?;
    writeln!(index_file, "    <p>Total Clusters: {}</p>", clusters.len())?;
    
    if !clusters.is_empty() {
        let total_nodes: usize = clusters.iter().map(|c| c.size).sum();
        let largest = clusters.iter().map(|c| c.size).max().unwrap_or(0);
        let avg_size = total_nodes as f64 / clusters.len() as f64;
        let avg_density = clusters.iter().map(|c| c.density as f64).sum::<f64>() / clusters.len() as f64;
        
        writeln!(index_file, "    <p>Total Clustered Nodes: {}</p>", total_nodes)?;
        writeln!(index_file, "    <p>Largest Cluster: {} nodes</p>", largest)?;
        writeln!(index_file, "    <p>Average Cluster Size: {:.2} nodes</p>", avg_size)?;
        writeln!(index_file, "    <p>Average Density: {:.4}</p>", avg_density)?;
    }
    
    writeln!(index_file, "  </div>")?;
    
    // Write cluster list
    writeln!(index_file, "  <h2>Clusters</h2>")?;
    writeln!(index_file, "  <div class=\"cluster-list\">")?;
    
    for cluster in clusters.iter().take(50) { // Limit to top 50 clusters
        writeln!(index_file, "    <div class=\"cluster-card\">")?;
        writeln!(index_file, "      <h3>Cluster {}</h3>", cluster.id)?;
        writeln!(index_file, "      <p>Size: {} nodes</p>", cluster.size)?;
        writeln!(index_file, "      <p>Density: {:.4}</p>", cluster.density)?;
        
        if !cluster.central_nodes.degree.is_empty() {
            writeln!(index_file, "      <p>Central Nodes: {} total</p>", cluster.central_nodes.degree.len())?;
        }
        
        writeln!(index_file, "    </div>")?;
    }
    
    writeln!(index_file, "  </div>")?;
    writeln!(index_file, "</body>")?;
    writeln!(index_file, "</html>")?;
    
    Ok(())
}

/// Generate statistical visualizations
fn generate_stats_visualizations(
    clusters: &[Cluster], 
    viz_dir: &Path
) -> Result<()> {
    log::info!("Generating statistical visualizations");
    
    // Create a CSV file with cluster statistics for external visualization
    let stats_path = viz_dir.join("cluster_stats.csv");
    let mut stats_file = File::create(stats_path)?;
    
    // Write header
    writeln!(stats_file, "cluster_id,size,density,central_nodes_count")?;
    
    // Write data for each cluster
    for cluster in clusters {
        writeln!(stats_file, "{},{},{:.6},{}", 
                 cluster.id, cluster.size, cluster.density, cluster.central_nodes.degree.len())?;
    }
    
    // Create a data file for size distribution
    let size_dist_path = viz_dir.join("size_distribution.csv");
    let mut size_dist_file = File::create(size_dist_path)?;
    
    // Create size distribution buckets
    let mut size_dist = vec![0; 11]; // 0: 1-9, 1: 10-19, 2: 20-29, ..., 9: 90-99, 10: 100+
    
    for cluster in clusters {
        let bucket = if cluster.size >= 100 {
            10
        } else {
            (cluster.size - 1) / 10  // 1-9 -> 0, 10-19 -> 1, etc.
        };
        
        size_dist[bucket] += 1;
    }
    
    // Write size distribution
    writeln!(size_dist_file, "size_range,count")?;
    writeln!(size_dist_file, "1-9,{}", size_dist[0])?;
    for i in 1..10 {
        let range_start = i * 10;
        let range_end = range_start + 9;
        writeln!(size_dist_file, "{}-{},{}", range_start, range_end, size_dist[i])?;
    }
    writeln!(size_dist_file, "100+,{}", size_dist[10])?;
    
    Ok(())
}
