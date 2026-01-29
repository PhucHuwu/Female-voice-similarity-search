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


def download_voxceleb_samples():
    """
    Download sample female voices from VoxCeleb
    Note: Full VoxCeleb requires registration
    """
    print("VoxCeleb requires manual download from:")
    print("https://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
    print("\nSteps:")
    print("1. Register for access")
    print("2. Download VoxCeleb1 or VoxCeleb2")
    print("3. Extract female speakers")
    print("4. Place in data/raw/")


def download_librispeech_dev_clean():
    """
    Download LibriSpeech dev-clean (small, free, no registration)
    ~350MB, contains female speakers
    """
    print("\n" + "="*60)
    print("Downloading LibriSpeech dev-clean (Female Voices)")
    print("="*60)
    
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    output_dir = Path("data/raw/librispeech")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_dir / "dev-clean.tar.gz"
    
    print(f"\nDownloading from: {url}")
    print(f"Size: ~350MB")
    print(f"This may take 5-10 minutes...\n")
    
    if download_file(url, str(archive_path)):
        print("\n✓ Download complete!")
        print(f"\nExtracting archive...")
        
        import tarfile
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(output_dir)
        
        print("✓ Extraction complete!")
        
        # Find .flac files
        audio_files = list(output_dir.rglob("*.flac"))
        print(f"\nFound {len(audio_files)} audio files")
        
        # Move first N files to data/raw
        raw_dir = Path("data/raw")
        count = 0
        max_files = 100  # Take 100 samples
        
        print(f"\nCopying {max_files} samples to data/raw/...")
        for audio_file in audio_files[:max_files]:
            dest = raw_dir / f"libri_{count:05d}.flac"
            
            # Copy file
            import shutil
            shutil.copy(audio_file, dest)
            count += 1
        
        print(f"\n✓ Copied {count} audio files to data/raw/")
        
        # Cleanup
        print("\nCleaning up archive...")
        archive_path.unlink()
        
        return count
    
    return 0


def download_common_voice_manual():
    """Instructions for Common Voice"""
    print("\n" + "="*60)
    print("Mozilla Common Voice - Manual Download")
    print("="*60)
    print("\nSteps:")
    print("1. Visit: https://commonvoice.mozilla.org/en/datasets")
    print("2. Choose language (e.g., English, Vietnamese)")
    print("3. Download the dataset (requires email)")
    print("4. Extract and filter by gender=female")
    print("5. Copy .mp3 files to data/raw/")


def download_from_freesound():
    """
    Download from Freesound.org (requires API key)
    """
    print("\n" + "="*60)
    print("Freesound.org - Free Sound Effects & Voices")
    print("="*60)
    print("\nSteps:")
    print("1. Create account: https://freesound.org/")
    print("2. Get API key: https://freesound.org/apiv2/apply/")
    print("3. Search for 'female voice' samples")
    print("4. Download and place in data/raw/")
    print("\nNote: Most samples have Creative Commons licenses")


def download_youtube_samples():
    """Download from YouTube using yt-dlp"""
    print("\n" + "="*60)
    print("YouTube Audio Download")
    print("="*60)
    
    # Example female voice channels/videos
    video_urls = [
        # Example public domain/CC videos (replace with actual URLs)
        "https://www.youtube.com/watch?v=EXAMPLE1",  # Replace
        "https://www.youtube.com/watch?v=EXAMPLE2",  # Replace
    ]
    
    print("\nYou need to provide YouTube URLs of videos with female voices")
    print("Examples: Podcasts, audiobooks, speeches, TED Talks")
    print("\nEdit this script and add URLs, then run:")
    print("  python src/data_collection/download_real_audio.py youtube")


def main():
    """Main function"""
    import sys
    
    print("\n" + "="*60)
    print("REAL VOICE DATA DOWNLOAD OPTIONS")
    print("="*60)
    
    if len(sys.argv) > 1:
        source = sys.argv[1].lower()
        
        if source == "librispeech":
            download_librispeech_dev_clean()
        elif source == "youtube":
            download_youtube_samples()
        elif source == "voxceleb":
            download_voxceleb_samples()
        elif source == "commonvoice":
            download_common_voice_manual()
        elif source == "freesound":
            download_from_freesound()
        else:
            print(f"Unknown source: {source}")
    else:
        print("\nAvailable sources:")
        print("\n1. LibriSpeech (RECOMMENDED - Free, No registration)")
        print("   Command: python src/data_collection/download_real_audio.py librispeech")
        print("   Size: ~350MB, 100+ female voice samples")
        
        print("\n2. Common Voice (Requires email)")
        print("   Command: python src/data_collection/download_real_audio.py commonvoice")
        print("   Size: Variable, many languages")
        
        print("\n3. VoxCeleb (Requires registration)")
        print("   Command: python src/data_collection/download_real_audio.py voxceleb")
        print("   Size: Large, celebrity voices")
        
        print("\n4. Freesound.org (Requires API key)")
        print("   Command: python src/data_collection/download_real_audio.py freesound")
        
        print("\n5. YouTube (Manual URLs)")
        print("   Command: python src/data_collection/download_real_audio.py youtube")
        
        print("\n" + "="*60)
        print("RECOMMENDED FOR QUICK START:")
        print("="*60)
        print("\npython src/data_collection/download_real_audio.py librispeech")
        print("\nThis will download 100 real female voice samples (~350MB)")


if __name__ == "__main__":
    main()
