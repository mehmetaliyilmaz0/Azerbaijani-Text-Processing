"""
Azerbaijani Sentiment Analysis Pipeline
Module: collect_youtube_data.py
Description:
    This script orchestrates the large-scale collection of YouTube comments for 
    the Azerbaijani Sentiment Analysis project. It features:
    
    1. Domain-Specific Discovery: Utilizes a curated keyword pool to target 
       the 5 required domains (Finance, Tech, etc.).
    2. Deep Pagination: Iterates through search result pages to ensure 
       diverse video sourcing.
    3. Linguistic Filtering: Applies the strict Azerbaijani filter (`part2_utils`) 
       at the comment level to reject Turkish/mixed content.
    4. Compliance Export: Saves data in the mandatory Excel format (A1 URL, A:Domain, B:Comment)
       with one file per video.
"""
import os
import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from part2_utils import DOMAINS, is_azerbaijani

# API CONFIGURATION
API_KEY = os.getenv("YOUTUBE_API_KEY")
OUTPUT_ROOT = "part2_data"

# GOALS
TARGET_COMMENTS_PER_DOMAIN = 10000 # The hard requirement
MAX_VIDEOS_TO_SEARCH = 500 # Safety cap to stop infinite loops if content is dry
MAX_COMMENTS_PER_VIDEO = 2000 # Increased to get more data from popular videos
MIN_CLEAN_COMMENTS = 5 # Lowered threshold to accept smaller contributions

# Keywords from project (Expanded for broader reach)
DOMAIN_KEYWORDS = {
    "Technology & Digital Services": [
        "telefon qiymətləri bakı", "honor baku", "xiaomi azərbaycan", "samsung qiymətləri", "iphone 15 baku",
        "internet paketləri", "yeni tətbiqlər", "oyunlar haqqında", "texnologiya xəbərləri az", "rəqəmsal marketinq bakı",
        "kompüter qiymətləri", "playstation 5 baku", "smart saatlar", "texno blog", "irşad electronics telefonlar",
        "kontakt home telefonlar", "bakcell tarifləri", "azercell internet", "nar mobile nömrələr", "notebook baku",
        "macbook qiymətləri", "redmi note 13", "samsung s24 ultra", "airpods baku", "oyun kompüterləri"
    ],
    "Finance & Business": [
        "biznes qurmaq bakı", "bank kreditləri", "kapital bank", "beynəlxalq bank", "abb mobile",
        "investisiya imkanları", "manat dollar məzənnəsi", "kredit faizləri", "sığorta şirkətləri",
        "sahibkarlıq", "vergi xəbərləri", "maaş artımı", "kriptovalyuta azərbaycan", "pasha bank",
        "unibank kredit", "leobank kart", "birbank keşbek", "ipoteka kreditləri", "ev krediti",
        "biznes kreditləri", "muhasibatlıq", "kassa aparatı", "vergi bəyannaməsi", "bitkoin azərbaycan"
    ],
    "Social Life & Entertainment": [
        "yeni mahnilar 2024", "meyxana", "toylarimiz", "azerbaycan filmleri", "baku tv",
        "yerli seriallar", "bakı vlog", "komediya səhnələri", "şou biznes", "məşhurlar",
        "konsert bakı", "trending azərbaycan", "yeni klip", "vine azərbaycan", "pərvin abıyeva",
        "röya ayxan", "zamiq hüseynov", "xezer tv", "atv xəbərlər", "arbitrum", "tiktok azərbaycan",
        "dadlı yeməklər", "restoranlar bakı", "əyləncə mərkəzləri"
    ],
    "Retail & Lifestyle": [
        "bazar qiymətləri bakı", "28 mall", "ganclik mall", "geyim mağazaları", "qızıl qiymətləri",
        "endirimler", "market qiymətləri", "qiymət artımı", "bravo market", "arazon", 
        "trendyol azərbaycan", "sədərək ticarət mərkəzi", "binə ticarət mərkəzi", "deniz mall",
        "gənclik mall mağazalar", "park bulvar", "zara baku", "lc waikiki baku", "ikinci əl geyim",
        "maşın bazarı bakı", "ev əşyaları", "mebel qiymətləri", "baku electronics endirim", "music gallery"
    ],
    "Public Services": [
        "asan xidmət", "təhsil nazirliyi", "sosial müdafiə fondu", "dövlət imtahan mərkəzi",
        "kommunal borc", "su pulu", "işiq pulu", "qaz pulu", "pensiya artımı", "tələbə krediti",
        "icbari tibbi sığorta", "uşaq pulu", "ünvanlı sosial yardım", "poliklinika", "xəstəxanalar",
        "bakı nəqliyyat agentliyi", "metro kartı", "azəriqaz", "azərişıq", "Azərsu", "xarici pasport",
        "şəxsiyyət vəsiqəsi", "ASAN login", "e-gov.az", "mygov az"
    ]
}

# =================================================================
# YOUTUBE API WRAPPER
# =================================================================

def get_youtube_service():
    """
    Initializes the Google API Client for YouTube Data v3.
    Returns:
        Resource: Authenticated YouTube service object.
    """
    return build('youtube', 'v3', developerKey=API_KEY)

def search_videos(youtube, query, max_results=20, pageToken=None):
    """
    Searches for videos matching the query.
    """
    print(f"  Searching for: '{query}'...")
    videos = []
    try:
        search_args = {
            'q': query,
            'type': 'video',
            'part': 'id,snippet',
            'maxResults': max_results,
            'relevanceLanguage': 'az', # Hint for Azerbaijani content
            'regionCode': 'AZ'         # Hint for region
        }
        if pageToken:
            search_args['pageToken'] = pageToken
            
        search_response = youtube.search().list(**search_args).execute()

        for search_result in search_response.get('items', []):
            if 'videoId' not in search_result['id']:
                continue
                
            vid = {
                'video_id': search_result['id']['videoId'],
                'title': search_result['snippet']['title'],
                'description': search_result['snippet']['description'],
                'channel': search_result['snippet']['channelTitle'],
                'publishedAt': search_result['snippet']['publishedAt']
            }
            videos.append(vid)
            
        return videos, search_response.get('nextPageToken')
            
    except HttpError as e:
        print(f"    An HTTP error {e.resp.status} occurred:\n    {e.content}")
        return [], None

def get_video_details(youtube, video_id):
    """
    Fetches detailed metadata (stats, tags, lang).
    """
    try:
        response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()
        
        if not response['items']:
            return None
            
        item = response['items'][0]
        snippet = item['snippet']
        stats = item['statistics']
        
        return {
            'tags': snippet.get('tags', []),
            'categoryId': snippet.get('categoryId'),
            'defaultLanguage': snippet.get('defaultLanguage'),
            'defaultAudioLanguage': snippet.get('defaultAudioLanguage'),
            'viewCount': stats.get('viewCount'),
            'likeCount': stats.get('likeCount'),
            'commentCount': stats.get('commentCount')
        }
    except Exception:
        return None

def get_comments(youtube, video_id, max_comments=200):
    """
    Fetches top-level comments for a video.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100, # Fetch max allowed per page to minimize API quota usage
            textFormat="plainText"
        )
        
        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response.get("items", []):
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment_text)
            
            # Pagination
            request = youtube.commentThreads().list_next(request, response)
            
    except HttpError as e: 
        if e.resp.status == 403: # Disabled comments
            return []
        print(f"    Error fetching comments: {e}")
    
    return comments

# =================================================================
# MAIN PIPELINE
# =================================================================

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    youtube = get_youtube_service()
    video_registry = set()

    for domain_name in DOMAINS:
        print(f"\n=== Processing Domain: {domain_name} ===")
        domain_dir = os.path.join(OUTPUT_ROOT, domain_name)
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)
            
        total_az_comments_collected = 0
        
        # Check existing files to count progress
        existing_files = [f for f in os.listdir(domain_dir) if f.endswith('.xlsx')]
        print(f"  Scanning existing {len(existing_files)} files to update count...")
        for f in existing_files:
            try:
                # Assuming correct format, reading just to count rows
                df = pd.read_excel(os.path.join(domain_dir, f), header=1)
                total_az_comments_collected += len(df)
            except Exception as e:
                print(f"    Warning: Could not read {f}: {e}")
        
        print(f"  Current Status: {total_az_comments_collected}/{TARGET_COMMENTS_PER_DOMAIN} comments.")
        
        if total_az_comments_collected >= TARGET_COMMENTS_PER_DOMAIN:
            print("  Target reached! Skipping domain.")
            continue

        keywords = DOMAIN_KEYWORDS.get(domain_name, [])
        videos_processed_count = 0
        
        for kw in keywords:
            if total_az_comments_collected >= TARGET_COMMENTS_PER_DOMAIN:
                break
                
            print(f"  Search Keyword: '{kw}'")
            
            # Pagination for search results (Go deeper)
            next_page_token = None
            pages_searched = 0
            
            while pages_searched < 5: # Search up to 5 pages per keyword (~250 videos potential)
                if total_az_comments_collected >= TARGET_COMMENTS_PER_DOMAIN:
                    break

                try:
                    # Search
                    # search_videos returns (videos_list, next_token)
                    results, next_page_token = search_videos(youtube, kw, max_results=50, pageToken=next_page_token)
                    
                    if not results:
                        break
                        
                    for vid_info in results:
                        if total_az_comments_collected >= TARGET_COMMENTS_PER_DOMAIN:
                            break
                            
                        vid_id = vid_info['video_id']
                        title = vid_info['title']
                        
                        if vid_id in video_registry or os.path.exists(os.path.join(domain_dir, f"{vid_id}.xlsx")):
                            continue
                            
                        print(f"    [{total_az_comments_collected}/{TARGET_COMMENTS_PER_DOMAIN}] Processing: {title[:40]}...")
                        
                        raw_comments = get_comments(youtube, vid_id, MAX_COMMENTS_PER_VIDEO)
                        if not raw_comments:
                            video_registry.add(vid_id)
                            continue

                        az_comments = []
                        for c in raw_comments:
                            c_clean = c.replace("\n", " ").strip()
                            if is_azerbaijani(c_clean):
                                az_comments.append(c_clean)
                        
                        # Skip if no Azerbaijani comments found
                        if len(az_comments) < MIN_CLEAN_COMMENTS:
                            if len(az_comments) == 0:
                                video_registry.add(vid_id)
                                continue

                        # SAVE
                        data = {"domain": [domain_name] * len(az_comments), "comment": az_comments}
                        df = pd.DataFrame(data)
                        out_path = os.path.join(domain_dir, f"{vid_id}.xlsx")
                        url = f"https://www.youtube.com/watch?v={vid_id}"
                        
                        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, startrow=1)
                            # Requirement: Cell A1 must contain the Video URL
                            writer.sheets['Sheet1']['A1'] = url
                            
                        print(f"      + Added {len(az_comments)} AZ comments.")
                        total_az_comments_collected += len(az_comments)
                        video_registry.add(vid_id)
                        videos_processed_count += 1
                    
                    if not next_page_token:
                        break
                    pages_searched += 1
                    
                except Exception as e:
                    print(f"    Search error: {e}")
                    break

    print("\n\nData Collection Complete!")

if __name__ == "__main__":
    main()
