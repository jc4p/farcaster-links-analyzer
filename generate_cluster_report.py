#!/usr/bin/env python3

import json
import os
import glob

def load_cluster_data():
    """Load cluster information and themes."""
    # Load theme summary
    try:
        with open('cluster_themes_summary.json', 'r') as f:
            themes = json.load(f)
    except FileNotFoundError:
        print("cluster_themes_summary.json not found. Please run identify_clusters.py first.")
        return None
    
    # Load original cluster data files to get size information
    clusters = []
    cluster_files = glob.glob('cluster_results/clusters/*.json')
    
    theme_dict = {item['cluster_id']: item['theme'] for item in themes}
    
    for file_path in cluster_files:
        try:
            with open(file_path, 'r') as f:
                cluster_data = json.load(f)
                cluster_id = cluster_data['id']
                
                # Add theme information if available
                if cluster_id in theme_dict:
                    theme_info = theme_dict[cluster_id]
                    cluster_data['theme'] = theme_info
                
                clusters.append(cluster_data)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    # Sort clusters by size (number of members)
    clusters.sort(key=lambda x: len(x.get('members', [])), reverse=True)
    
    return clusters

def generate_markdown_report(clusters):
    """Generate a concise Markdown report of clusters."""
    if not clusters:
        return
    
    md_content = "# Farcaster Network Cluster Analysis\n\n"
    md_content += "## Clusters by Size\n\n"
    
    # Table of contents
    md_content += "| Rank | Size | Cluster Name | Density |\n"
    md_content += "|------|------|-------------|--------|\n"
    
    for i, cluster in enumerate(clusters):
        cluster_id = cluster['id']
        size = len(cluster.get('members', []))
        density = cluster.get('density', 'N/A')
        
        # Get cluster name from theme if available
        if 'theme' in cluster and 'cluster_name' in cluster['theme']:
            name = cluster['theme']['cluster_name']
        else:
            name = f"Cluster {cluster_id}"
        
        md_content += f"| {i+1} | {size} | [{name}](#{name.lower().replace(' ', '-')}) | {density:.4f} |\n"
    
    # Detailed cluster descriptions
    md_content += "\n## Cluster Details\n\n"
    
    for cluster in clusters:
        cluster_id = cluster['id']
        size = len(cluster.get('members', []))
        density = cluster.get('density', 'N/A')
        
        # Get theme information if available
        if 'theme' in cluster:
            theme = cluster['theme']
            name = theme.get('cluster_name', f"Cluster {cluster_id}")
            common_themes = theme.get('common_themes', [])
            description = theme.get('description', 'No description available.')
            
            # Format common themes as a bulleted list if it's a list
            if isinstance(common_themes, list):
                themes_formatted = "\n".join([f"- {t}" for t in common_themes])
            else:
                themes_formatted = common_themes
        else:
            name = f"Cluster {cluster_id}"
            themes_formatted = "No theme information available."
            description = ""
        
        md_content += f"### {name}\n\n"
        md_content += f"**ID:** {cluster_id} | **Size:** {size} members | **Density:** {density:.4f}\n\n"
        
        if 'theme' in cluster:
            md_content += "**Common Themes:**\n"
            md_content += f"{themes_formatted}\n\n"
            md_content += f"**Description:** {description}\n\n"
        
        # Add central nodes information
        if 'central_nodes' in cluster:
            md_content += "**Central Nodes:**\n"
            for metric, nodes in cluster['central_nodes'].items():
                top_nodes = nodes[:5]
                md_content += f"- {metric}: {', '.join(top_nodes)}"
                if len(nodes) > 5:
                    md_content += f" (+ {len(nodes) - 5} more)"
                md_content += "\n"
            md_content += "\n"
    
    return md_content

def main():
    print("Loading cluster data...")
    clusters = load_cluster_data()
    
    if not clusters:
        return
    
    print(f"Generating Markdown report for {len(clusters)} clusters...")
    md_content = generate_markdown_report(clusters)
    
    output_file = "farcaster_clusters.md"
    with open(output_file, 'w') as f:
        f.write(md_content)
    
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    main()