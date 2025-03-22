#!/usr/bin/env python3

import json
import os
import glob
import time
from google import genai
from google.genai import types
import dotenv

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def load_cluster_center_points():
    """Load all cluster center points JSON files."""
    center_points_files = glob.glob('cluster_*_center_points.json')
    
    if not center_points_files:
        print("No cluster center points files found. Please run analyze_clusters.py first.")
        return []
    
    # Sort by cluster ID
    center_points_files.sort(key=lambda x: int(x.split('_')[1]))
    
    cluster_data = []
    for file_path in center_points_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                cluster_id = int(file_path.split('_')[1])
                cluster_data.append({
                    'id': cluster_id,
                    'file': file_path,
                    'data': data
                })
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return cluster_data

def extract_user_info(user_data):
    """Extract relevant user information from the API response."""
    extracted_info = {
        'fid': user_data.get('fid'),
        'bio': None,
        'casts': []
    }
    
    # Extract bio and casts
    if 'data' in user_data and 'casts' in user_data['data']:
        casts = user_data['data']['casts']
        
        # Get bio from the first cast's author profile if available
        if casts and 'author' in casts[0] and 'profile' in casts[0]['author'] and 'bio' in casts[0]['author']['profile']:
            bio_data = casts[0]['author']['profile']['bio']
            if 'text' in bio_data:
                extracted_info['bio'] = bio_data['text']
        
        # Extract cast texts
        for cast in casts:
            if 'text' in cast:
                extracted_info['casts'].append(cast['text'])
    
    return extracted_info

def identify_cluster_theme(cluster_data):
    """Use Gemini API to identify the theme of a cluster based on center points data."""
    cluster_id = cluster_data['id']
    print(f"\nAnalyzing Cluster {cluster_id}...")
    
    # Extract relevant user information
    users_info = []
    for user_data in cluster_data['data']:
        user_info = extract_user_info(user_data)
        users_info.append(user_info)
    
    # Prepare prompt for Gemini
    bios = [f"User {user['fid']} Bio: {user['bio']}" for user in users_info if user['bio']]
    
    # Limit the number of casts to avoid making the prompt too large
    max_casts_per_user = 5
    all_casts = []
    for user in users_info:
        casts = user['casts'][:max_casts_per_user]
        all_casts.extend([f"User {user['fid']} Cast: {cast}" for cast in casts])
    
    # Sample casts if there are too many
    max_casts = 30
    if len(all_casts) > max_casts:
        import random
        all_casts = random.sample(all_casts, max_casts)
    
    bios_text = "\n".join(bios)
    casts_text = "\n".join(all_casts)
    
    prompt_text = f"""
You are analyzing a social media cluster on Farcaster. Below are bios and sample posts from central users in the cluster.

BIOS:
{bios_text}

SAMPLE POSTS:
{casts_text}

Based on these users' bios and posts, what would be a concise, descriptive name for this cluster? 
What common interests, identities, topics, or behaviors do these users share?
Provide your analysis in JSON format with these fields:
- cluster_name: A short, descriptive name for this cluster (3-5 words)
- common_themes: List the main themes, interests or topics shared by these users
- description: A brief explanation of what unites this cluster (2-3 sentences)
"""

    # Call Gemini API
    try:
        # Initialize the Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Create content and config
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
            response_mime_type="application/json",
        )
        
        print(f"  Calling Gemini API...")
        
        # Get a response from Gemini (non-streaming)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=generate_content_config,
        )
        
        text = response.text
        
        # Extract JSON from text
        import re
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        try:
            result = json.loads(text)
            
            # Save result to file
            output_file = f"cluster_{cluster_id}_theme.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"  Cluster theme: {result.get('cluster_name', 'Unknown')}")
            print(f"  Saved theme to {output_file}")
            
            return result
        except json.JSONDecodeError:
            print(f"  Error parsing JSON response, saving raw response instead")
            output_file = f"cluster_{cluster_id}_theme_raw.txt"
            with open(output_file, 'w') as f:
                f.write(text)
            print(f"  Saved raw response to {output_file}")
            
            return {"cluster_name": "Could not parse", "raw_response": text}
    
    except Exception as e:
        print(f"  Error calling Gemini API: {str(e)}")
        return {"cluster_name": "Error", "error": str(e)}

def main():
    print("Loading cluster center points...")
    clusters = load_cluster_center_points()
    print(f"Found {len(clusters)} cluster data files.")
    
    if not clusters:
        return
    
    cluster_themes = []
    
    for cluster in clusters:
        theme = identify_cluster_theme(cluster)
        cluster_themes.append({
            'cluster_id': cluster['id'],
            'theme': theme
        })
        
        # Rate limiting
        time.sleep(1)
    
    # Save summary of all themes
    with open('cluster_themes_summary.json', 'w') as f:
        json.dump(cluster_themes, f, indent=2)
    
    print("\nCluster Themes Summary:")
    for item in cluster_themes:
        cluster_id = item['cluster_id']
        theme = item['theme']
        name = theme.get('cluster_name', 'Unknown')
        print(f"Cluster {cluster_id}: {name}")
    
    print("\nSaved all themes to cluster_themes_summary.json")

if __name__ == "__main__":
    main()