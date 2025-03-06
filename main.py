#!/usr/bin/env python3
"""
Script for downloading transcripts from a YouTube channel and creating a JSONL training dataset
Obtain an API key for YouTube Data API v3 (see instructions in README.md).
"""

import json
import re
import os
import sys
import random
import argparse
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcript-parser")

# Load environment variables from .env file
load_dotenv()

def extract_channel_id(channel_url):
    """
    Function to extract the channel ID from a URL.
    Supports format: https://www.youtube.com/channel/CHANNEL_ID
    """
    match = re.search(r"youtube\.com/channel/([^/?]+)", channel_url)
    return match.group(1) if match else None

def get_videos_from_channel(api_key: str, channel_id: str) -> List[Dict[str, str]]:
    """
    Gets a list of videos from a channel using YouTube Data API v3.
    Returns a list of dictionaries with video_id, title and publication date.
    
    Args:
        api_key: YouTube Data API key
        channel_id: YouTube channel ID
        
    Returns:
        List of dictionaries containing video information
        
    Raises:
        HttpError: If there's an API error
    """
    logger.info(f"Fetching videos for channel: {channel_id}")
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        videos = []
        nextPageToken = None
        page_count = 0

        while True:
            page_count += 1
            logger.debug(f"Fetching page {page_count} of results")
            
            try:
                request = youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    maxResults=50,
                    order="date",  # returns videos from newest to oldest
                    pageToken=nextPageToken,
                    type="video"
                )
                response = request.execute()

                for item in response.get("items", []):
                    video_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    published_at = item["snippet"]["publishedAt"]
                    videos.append({
                        "video_id": video_id,
                        "title": title,
                        "published_at": published_at
                    })
                    
                nextPageToken = response.get("nextPageToken")
                if not nextPageToken:
                    break
                    
            except HttpError as e:
                logger.error(f"YouTube API error: {e}")
                if e.resp.status == 403:
                    logger.error("API quota may be exhausted or key is invalid")
                raise
                
        # Sort by publication date from oldest to newest
        videos.sort(key=lambda x: x["published_at"])
        logger.info(f"Found {len(videos)} videos")
        return videos
        
    except Exception as e:
        logger.error(f"Error fetching videos: {str(e)}")
        raise

def get_transcript_for_video(video_id: str, language_code: str) -> Optional[str]:
    """
    Attempts to get a transcript for a video.
    First tries to get a transcript in the specified language,
    if unsuccessful, tries an automatically generated version (with "a." prefix).
    
    Args:
        video_id: YouTube video ID
        language_code: Language code for transcripts
        
    Returns:
        String with the transcript text or None if the transcript is unavailable.
    """
    logger.debug(f"Getting transcript for video ID: {video_id}")
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        logger.debug(f"Found transcript in language: {language_code}")
        transcript_text = " ".join([seg["text"] for seg in transcript])
        return transcript_text
        
    except NoTranscriptFound:
        logger.debug(f"No transcript found in {language_code}, trying auto-generated")
        # Try to get automatically generated transcript
        try:
            auto_lang = "a." + language_code
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[auto_lang])
            logger.debug(f"Found auto-generated transcript in language: {language_code}")
            transcript_text = " ".join([seg["text"] for seg in transcript])
            return transcript_text
            
        except NoTranscriptFound:
            logger.debug(f"No auto-generated transcript found for video {video_id}")
            return None
            
        except TranscriptsDisabled:
            logger.debug(f"Transcripts are disabled for video {video_id}")
            return None
            
        except Exception as e:
            logger.debug(f"Error getting auto-generated transcript: {str(e)}")
            return None
            
    except TranscriptsDisabled:
        logger.debug(f"Transcripts are disabled for video {video_id}")
        return None
        
    except Exception as e:
        logger.debug(f"Error getting transcript: {str(e)}")
        return None

def preprocess_transcript(transcript: str) -> str:
    """
    Clean and preprocess the transcript text.
    """
    if not transcript:
        return ""
    
    # Remove excessive whitespace
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    
    # Add additional preprocessing logic as needed
    
    return transcript

def create_dataset_entry(video_title: str, published_at: str, transcript: str) -> Dict[str, Any]:
    """
    Create a dataset entry in the new requested format.
    """
    return {
        "created": published_at,
        "name": video_title,
        "value": transcript if transcript else "no transcript available"
    }

def process_videos_streaming(videos: List[Dict], language_code: str, output_file: str) -> int:
    """
    Process videos one by one and write each processed transcript directly to a JSONL file.
    
    Args:
        videos: List of video dictionaries with video_id, title and published_at
        language_code: Language code for transcripts
        output_file: Path to the output JSONL file
        
    Returns:
        Count of successfully processed videos
    """
    count = 0
    total = len(videos)
    
    logger.info(f"Starting to process {total} videos")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Use tqdm to create a progress bar
            for video in tqdm(videos, desc="Processing videos", unit="video"):
                try:
                    video_id = video['video_id']
                    title = video['title']
                    
                    # Log at debug level so it doesn't clutter the progress bar
                    logger.debug(f"Processing video: {title} (ID: {video_id})")
                    
                    transcript = get_transcript_for_video(video_id, language_code)
                    
                    if transcript:
                        # Preprocess transcript
                        processed_transcript = preprocess_transcript(transcript)
                        logger.debug(f"Transcript processed: {len(processed_transcript)} characters")
                    else:
                        processed_transcript = None
                        logger.debug("No transcript available")
                        
                    # Create dataset entry
                    entry = create_dataset_entry(
                        video_title=title,
                        published_at=video["published_at"],
                        transcript=processed_transcript
                    )
                    
                    # Write directly to file
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing video {video['video_id']}: {str(e)}")
                    # Continue with the next video
                    
    except IOError as e:
        logger.error(f"I/O error writing to {output_file}: {str(e)}")
        raise
        
    logger.info(f"Successfully processed {count} out of {total} videos")
    return count

def sample_jsonl_file(input_file: str, output_file: str, ratio: float) -> int:
    """
    Sample entries from a JSONL file based on the given ratio and write to a new file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        ratio: Sampling ratio (0.0 to 1.0)
        
    Returns:
        Number of entries written
        
    Raises:
        ValueError: If ratio is invalid
        IOError: If file operations fail
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Sampling ratio must be between 0 and 1")
        
    logger.info(f"Sampling {ratio:.1%} of entries from {input_file}")
    
    try:
        # First, count total lines to know how many to sample
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        sample_size = int(total_lines * ratio)
        if sample_size == 0 and total_lines > 0:
            sample_size = 1  # Ensure at least one item if possible
        
        logger.debug(f"Total lines: {total_lines}, sample size: {sample_size}")
        
        # Get random line indices to sample
        if total_lines > 0:
            sample_indices = set(random.sample(range(total_lines), sample_size))
        else:
            sample_indices = set()
        
        # Read input file again and write selected lines to output
        count = 0
        with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'w', encoding='utf-8') as out_f:
            for i, line in enumerate(in_f):
                if i in sample_indices:
                    out_f.write(line)
                    count += 1
        
        logger.info(f"Wrote {count} entries to {output_file}")
        return count
        
    except IOError as e:
        logger.error(f"I/O error during sampling: {str(e)}")
        raise

def jsonl_to_json(jsonl_file: str, json_file: str) -> int:
    """
    Convert a JSONL file to a standard JSON array file.
    
    Args:
        jsonl_file: Path to input JSONL file
        json_file: Path to output JSON file
        
    Returns:
        Number of entries processed
        
    Raises:
        IOError: If file operations fail
        json.JSONDecodeError: If JSONL content is invalid
    """
    logger.info(f"Converting {jsonl_file} to JSON format")
    data = []
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {i+1}: {str(e)}")
                        raise
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Converted {len(data)} entries to {json_file}")
        return len(data)
        
    except IOError as e:
        logger.error(f"I/O error during JSON conversion: {str(e)}")
        raise

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download transcripts from YouTube channels and create training datasets."
    )
    parser.add_argument(
        "channel_url", 
        help="YouTube channel URL in format https://www.youtube.com/channel/CHANNEL_ID"
    )
    parser.add_argument(
        "language_code", 
        help="Language code for transcripts (e.g., 'en' for English)"
    )
    parser.add_argument(
        "--api-key", 
        help="YouTube Data API key (can also be set via YOUTUBE_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-dir", 
        default="dataset", 
        help="Directory to save output files (default: dataset)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """
    Main function to execute the transcript downloading process.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging level based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Get API key from command line or environment variable
    api_key = args.api_key or os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        logger.error("YouTube API key is required")
        logger.error("You can provide it with --api-key or set the YOUTUBE_API_KEY environment variable")
        sys.exit(1)
        
    try:
        channel_id = extract_channel_id(args.channel_url)
        if not channel_id:
            logger.error("Invalid channel URL format")
            logger.error("Please use format: https://www.youtube.com/channel/CHANNEL_ID")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error extracting channel ID: {str(e)}")
        sys.exit(1)

    try:
        # Get videos from channel
        videos = get_videos_from_channel(api_key, channel_id)
        
        if not videos:
            logger.warning("No videos found for this channel")
            sys.exit(0)

        # Create output directory if it doesn't exist
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Process videos and stream directly to JSONL file
        full_dataset_jsonl_path = os.path.join(output_dir, "full_dataset.jsonl")
        processed_count = process_videos_streaming(videos, args.language_code, full_dataset_jsonl_path)
        
        if processed_count == 0:
            logger.warning("No transcripts were successfully processed")
            sys.exit(0)
        
        # Check if we have enough data to split
        if processed_count < 3:
            logger.warning("Not enough data to create meaningful splits (minimum 3 entries required)")
        else:
            # Create train/val/test splits by sampling the full dataset
            logger.info("Creating train/val/test splits...")
            
            train_file = os.path.join(output_dir, "train.jsonl")
            val_file = os.path.join(output_dir, "val.jsonl")
            test_file = os.path.join(output_dir, "test.jsonl")
            
            # Sample 80% for train
            train_count = sample_jsonl_file(full_dataset_jsonl_path, train_file, 0.8)
            
            # Sample 10% for validation
            val_count = sample_jsonl_file(full_dataset_jsonl_path, val_file, 0.1)
            
            # Sample 10% for testing
            test_count = sample_jsonl_file(full_dataset_jsonl_path, test_file, 0.1)
        
        # Convert to JSON format
        json_path = os.path.join(output_dir, "transcripts.json")
        json_count = jsonl_to_json(full_dataset_jsonl_path, json_path)
        
        logger.info(f"Dataset creation complete. Files saved in the '{output_dir}' directory")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
