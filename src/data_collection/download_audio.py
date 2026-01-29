"""
Audio download tool for building voice dataset
Supports multiple sources: HuggingFace datasets, YouTube, local files
"""
import os
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from datasets import load_dataset
import yt_dlp


class AudioDownloader:
    """Download and manage audio files for voice similarity search"""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize audio downloader
        
        Args:
            output_dir: Directory to save downloaded audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = []
    
    def download_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        gender_filter: str = "female"
    ) -> int:
        """
        Download audio from HuggingFace datasets
        
        Args:
            dataset_name: Name of the dataset (e.g., "mozilla-foundation/common_voice_13_0")
            split: Dataset split to use
            max_samples: Maximum number of samples to download
            gender_filter: Filter by gender if available
            
        Returns:
            Number of files downloaded
        """
        print(f"Loading dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name, "vi", split=split, streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying without language specification...")
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        count = 0
        for idx, sample in enumerate(dataset):
            if max_samples and count >= max_samples:
                break
            
            # Filter by gender if field exists
            if gender_filter and 'gender' in sample:
                if sample['gender'] != gender_filter:
                    continue
            
            # Get audio data
            audio = sample.get('audio', None)
            if audio is None:
                continue
            
            # Save audio file
            file_name = f"hf_{count:05d}.wav"
            file_path = self.output_dir / file_name
            
            # Write audio to file
            import soundfile as sf
            sf.write(file_path, audio['array'], audio['sampling_rate'])
            
            # Add metadata
            self.metadata.append({
                'file_path': str(file_path),
                'source': dataset_name,
                'index': count,
                'original_sr': audio['sampling_rate'],
                'duration': len(audio['array']) / audio['sampling_rate']
            })
            
            count += 1
            if count % 10 == 0:
                print(f"Downloaded {count} files...")
        
        print(f"Total downloaded from HuggingFace: {count}")
        return count
    
    def download_from_youtube(
        self,
        url: str,
        max_duration: int = 300
    ) -> Optional[str]:
        """
        Download audio from YouTube video
        
        Args:
            url: YouTube video URL
            max_duration: Maximum video duration in seconds
            
        Returns:
            Path to downloaded file or None if failed
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'outtmpl': str(self.output_dir / 'yt_%(id)s.%(ext)s'),
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                
                if info['duration'] > max_duration:
                    print(f"Video too long: {info['duration']}s")
                    return None
                
                file_path = self.output_dir / f"yt_{info['id']}.wav"
                
                self.metadata.append({
                    'file_path': str(file_path),
                    'source': 'youtube',
                    'url': url,
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0)
                })
                
                print(f"Downloaded: {info.get('title', url)}")
                return str(file_path)
        
        except Exception as e:
            print(f"Error downloading from YouTube: {e}")
            return None
    
    def save_metadata(self, output_path: str = "data/metadata.csv") -> None:
        """
        Save metadata to CSV file
        
        Args:
            output_path: Path to save metadata CSV
        """
        df = pd.DataFrame(self.metadata)
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")
    
    def load_metadata(self, metadata_path: str = "data/metadata.csv") -> pd.DataFrame:
        """
        Load metadata from CSV file
        
        Args:
            metadata_path: Path to metadata CSV
            
        Returns:
            Metadata DataFrame
        """
        return pd.read_csv(metadata_path)


def download_sample_dataset(output_dir: str = "data/raw", num_samples: int = 50):
    """
    Download a sample dataset for testing
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to download
    """
    downloader = AudioDownloader(output_dir)
    
    print(f"Downloading {num_samples} sample audio files...")
    print("Source: Mozilla Common Voice (Vietnamese)")
    
    # Download from Common Voice
    count = downloader.download_from_huggingface(
        dataset_name="mozilla-foundation/common_voice_13_0",
        split="train",
        max_samples=num_samples,
        gender_filter="female"
    )
    
    # Save metadata
    downloader.save_metadata()
    
    print(f"\n✓ Downloaded {count} files to {output_dir}")
    print("✓ Metadata saved to data/metadata.csv")


if __name__ == "__main__":
    # Example usage
    download_sample_dataset(num_samples=50)
