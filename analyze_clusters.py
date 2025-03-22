#!/usr/bin/env python3

import json
import os
import glob
import subprocess
import time
import dotenv

dotenv.load_dotenv()

NEYNAR_API_KEY = os.getenv('NEYNAR_API_KEY')

def load_clusters():
    """Load all cluster JSON files from the clusters directory."""
    clusters = []
    cluster_files = glob.glob('cluster_results/clusters/*.json')
    
    for file_path in sorted(cluster_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
        with open(file_path, 'r') as f:
            cluster_data = json.load(f)
            clusters.append(cluster_data)
    
    return clusters

def analyze_clusters(clusters):
    """Analyze and print information about each cluster."""
    for cluster in clusters:
        cluster_id = cluster['id']
        members_count = len(cluster['members'])
        density = cluster.get('density', 'N/A')
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Members: {members_count}")
        print(f"  Density: {density}")
        
        print("  Central nodes:")
        for metric, nodes in cluster.get('central_nodes', {}).items():
            print(f"    {metric}: {', '.join(nodes[:5])}" + (f" (+ {len(nodes) - 5} more)" if len(nodes) > 5 else ""))
        
        # Fetch data for central nodes
        fetch_user_data_for_cluster(cluster)

def fetch_user_data_for_cluster(cluster):
    """Fetch Farcaster user data for central nodes in a cluster and save to JSON file."""
    cluster_id = cluster['id']
    output_file = f"cluster_{cluster_id}_center_points.json"
    
    # Get all central nodes across metrics
    central_nodes = set()
    for nodes in cluster.get('central_nodes', {}).values():
        central_nodes.update(nodes)
    
    print(f"  Fetching data for {len(central_nodes)} central nodes...")
    
    all_user_data = []
    
    for user_fid in central_nodes:
        print(f"    Fetching data for user {user_fid}...")
        try:
            # Call Neynar API
            api_url = f'https://api.neynar.com/v2/farcaster/feed/user/casts?fid={user_fid}&limit=150&include_replies=false'
            result = subprocess.run(
                ['curl', '--request', 'GET', 
                 '--url', api_url,
                 '--header', 'accept: application/json',
                 '--header', f'x-api-key: {NEYNAR_API_KEY}'],
                capture_output=True, text=True, check=True
            )
            
            # Parse JSON response
            user_data = json.loads(result.stdout)
            
            # Add to collection
            all_user_data.append({
                'fid': user_fid,
                'data': user_data
            })
            
            # Rate limiting - be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error fetching data for user {user_fid}: {str(e)}")
    
    # Save all data to file
    with open(output_file, 'w') as f:
        json.dump(all_user_data, f, indent=2)
    
    print(f"  Saved central node data to {output_file}")

def main():
    print("Loading clusters from cluster_results/clusters/*.json...")
    clusters = load_clusters()
    print(f"Found {len(clusters)} clusters.")
    
    analyze_clusters(clusters)

if __name__ == "__main__":
    main()