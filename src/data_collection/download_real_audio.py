"""
Download real female voice samples from public datasets
Multiple sources available
"""
import os
import requests
from pathlib import Path
import json
import zipfile
from tqdm import tqdm


def download_file(url: str, output_path: str) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(output_path)
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def download_youtube_samples():
    """Download from YouTube using yt-dlp with workarounds"""
    print("\n" + "="*60)
    print("YouTube Audio Download")
    print("="*60)
    
    try:
        import yt_dlp
    except ImportError:
        print("\nError: yt-dlp not installed")
        print("Install with: pip install yt-dlp")
        return 0
    
    # YouTube URLs to download
    video_urls = [
        "https://www.youtube.com/watch?v=SSGkkMEeoJE",  # 1
        "https://www.youtube.com/watch?v=NEZnRlVuKg8",  # 2
        "https://www.youtube.com/watch?v=qZuxop5xj_E",  # 3
        "https://www.youtube.com/watch?v=fBnjGCEmzGQ",  # 4
        "https://www.youtube.com/watch?v=9c-yi4vCqZg",  # 5
        "https://www.youtube.com/watch?v=pshSh--QiIo",  # 6
    ]
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {len(video_urls)} videos...")
    print(f"Output directory: {output_dir}\n")
    
    # Enhanced yt-dlp options with YouTube workarounds
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': str(output_dir / 'yt_%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        
        # YouTube-specific workarounds
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        
        # User agent spoofing
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        
        # Additional headers
        'http_headers': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.youtube.com/',
        },
        
        # Retry and timeout settings
        'retries': 3,
        'fragment_retries': 3,
        'socket_timeout': 30,
        
        # Skip unavailable formats
        'ignoreerrors': False,
        'no_color': False,
    }
    
    count = 0
    failed = []
    
    for idx, url in enumerate(video_urls, 1):
        try:
            print(f"\n[{idx}/{len(video_urls)}] Downloading: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info:
                    video_id = info.get('id', f'unknown_{idx}')
                    title = info.get('title', 'Unknown')
                    duration = info.get('duration', 0)
                    
                    print(f"Downloaded: {title} ({duration}s)")
                    count += 1
                else:
                    print(f"Failed: No info returned")
                    failed.append(url)
                
        except yt_dlp.utils.DownloadError as e:
            print(f"Download error: {str(e)[:100]}")
            failed.append(url)
        except Exception as e:
            print(f"Unexpected error: {str(e)[:100]}")
            failed.append(url)
    
    print("\n" + "="*60)
    print(f"Download complete: {count}/{len(video_urls)} successful")
    if failed:
        print(f"\nFailed URLs ({len(failed)}):")
        for url in failed:
            print(f"  - {url}")
        print("\nNote: If downloads fail, try:")
        print("1. Update yt-dlp: pip install --upgrade yt-dlp")
        print("2. Check if videos are available in your region")
        print("3. Use different video URLs")
    print("="*60)
    
    return count


def main():
    """Main function"""
    download_youtube_samples()

if __name__ == "__main__":
    main()
