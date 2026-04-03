"""
Azerbaijani Sentiment Analysis Pipeline
Module: fetch_metadata_retroactive.py
Description:
    This utility script is designed to retroactively fetch and recover metadata 
    for the collected YouTube dataset. It ensures data completeness by:
    1. Scanning the local `part2_data` directory for existing video IDs.
    2. Batch-querying the YouTube Data API (50 items/request) to retrieving missing stats.
    3. Generating a master `video_metadata_all.xlsx` file required for domain verification.
    
    The script serves as a data integrity layer, ensuring that even if initial 
    collection metadata was partial, the final submission includes comprehensive 
    stats (views, likes, tags) for every video.
"""
import os
import pandas as pd
from googleapiclient.discovery import build
from part2_utils import DOMAINS

# CONFIGURATION
input_dir = "part2_data"
output_file = "video_metadata_all.xlsx"

API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_youtube_service():
    """
    Initializes the authenticated YouTube Data API v3 service.
    """
    return build('youtube', 'v3', developerKey=API_KEY)

def get_video_stats(youtube, video_ids):
    """
    Retrieves detailed statistics for a list of video IDs using batch processing.
    
    Args:
        youtube: The authenticated API service instance.
        video_ids (list): List of YouTube video ID strings.
        
    Returns:
        tuple: (stats_map, failed_ids) - A mapping of video_id -> metadata and list of failed IDs.
        
    Implementation Note:
        The YouTube API enforces a limit of 50 IDs per request. 
        The function handles chunking automatically to respect this quota.
    """
    stats_map = {}
    failed_ids = []
    
    # API allows 50 ids per call - Chunking is strictly enforced.
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part='snippet,statistics',
                id=','.join(chunk)
            ).execute()
            
            # Track retrieved IDs
            retrieved_ids = set()
            for item in resp.get('items', []):
                vid = item['id']
                retrieved_ids.add(vid)
                snip = item['snippet']
                stat = item['statistics']
                stats_map[vid] = {
                    'video_id': vid,
                    'title': snip.get('title'),
                    'channel': snip.get('channelTitle'),
                    'publishedAt': snip.get('publishedAt'),
                    'tags': ",".join(snip.get('tags', [])),
                    'categoryId': snip.get('categoryId'),
                    'viewCount': stat.get('viewCount'),
                    'likeCount': stat.get('likeCount'),
                    'commentCount': stat.get('commentCount')
                }
            
            # Track missing IDs (deleted/private videos)
            for vid in chunk:
                if vid not in retrieved_ids:
                    failed_ids.append(vid)
                    
        except Exception as e:
            print(f"Error fetching chunk: {e}")
            failed_ids.extend(chunk)  # Mark entire chunk as failed
            
    return stats_map, failed_ids

def main():
    print("Scanning part2_data for videos...")
    video_ids = []
    domain_map = {}
    
    # Walk directory to find video IDs from filenames (ID.xlsx)
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".xlsx") and not f.startswith("~"):
                vid_id = os.path.splitext(f)[0]
                video_ids.append(vid_id)
                domain = os.path.basename(root)
                domain_map[vid_id] = domain
                
    unique_ids = list(set(video_ids))
    print(f"Found {len(unique_ids)} unique videos.")
    
    if not unique_ids:
        print("No videos found.")
        return

    youtube = get_youtube_service()
    print("Fetching metadata from YouTube API...")
    stats, failed_ids = get_video_stats(youtube, unique_ids)
    
    # Report failed IDs
    if failed_ids:
        print(f"WARNING: {len(failed_ids)} videos could not be retrieved (deleted/private).")
        print(f"Failed IDs: {failed_ids[:10]}...")  # Show first 10
    
    # Combine into DataFrame
    data = []
    for vid, info in stats.items():
        info['assigned_domain'] = domain_map.get(vid, "Unknown")
        data.append(info)
        
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Metadata saved to {output_file} ({len(data)} videos)")

if __name__ == "__main__":
    main()
