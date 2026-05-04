"""
Download real female voice samples from public datasets
Multiple sources available
"""
import os
import re
import requests
from pathlib import Path
import json
import zipfile
import unicodedata
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


def sanitize_filename_part(text: str) -> str:
    """Sanitize text for safe filename usage."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return cleaned or "unknown"


def load_video_entries(csv_path: Path) -> list[tuple[str, str]]:
    """Load (url, voice) entries from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    video_entries = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f):
            line = raw_line.strip()
            if i == 0 or not line:
                continue

            if line.startswith("http"):
                parts = line.split(",", 1)
                url = parts[0].strip().strip('"')
                voice = parts[1].strip().strip('"') if len(parts) > 1 else "unknown"
                if url:
                    video_entries.append((url, voice or "unknown"))

    return video_entries

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
    
    csv_path = Path("data/list_video.csv")
    try:
        video_entries = load_video_entries(csv_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 0

    if not video_entries:
        print(f"\nError: No video URLs found in {csv_path}")
        return 0
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoaded {len(video_entries)} video URLs from {csv_path}")
    print(f"Downloading {len(video_entries)} videos...")
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
    
    for idx, (url, voice) in enumerate(video_entries, 1):
        try:
            safe_voice = sanitize_filename_part(voice)
            ydl_opts_current = dict(ydl_opts)
            ydl_opts_current['outtmpl'] = str(output_dir / f'yt_{safe_voice}_%(id)s.%(ext)s')

            print(f"\n[{idx}/{len(video_entries)}] Downloading: {url}")
            
            with yt_dlp.YoutubeDL(ydl_opts_current) as ydl:
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
    print(f"Download complete: {count}/{len(video_entries)} successful")
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
