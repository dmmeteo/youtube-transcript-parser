# YouTube Transcript Scraper

A CLI tool for downloading transcripts for all YouTube videos at a specified channel and creating training datasets for language models.

## Instructions for obtaining an API key for YouTube Data API v3

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. In the left menu, navigate to APIs & Services → Library and find YouTube Data API v3.
4. Click Enable to activate the API for your project.
5. Navigate to APIs & Services → Credentials.
6. Click Create Credentials and select API key.
7. Copy the API key you received and use it when running the script.

## Setup

We use the `uv` package manager for dependency management. To set up the project:

1. Create a virtual environment:
```bash
uv venv --python 3.13
```

2. Install dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## Running the script
```bash
python main.py "https://www.youtube.com/channel/UCXXXXXXX" en $YOUR_API_KEY
```

## Environment Variables
You can also provide your API key using environment variables or a .env file:

1. Create a .env file in the project directory
2. Add your YouTube API key: `YOUTUBE_API_KEY=your_key_here`

When using environment variables, you can run the script without providing the API key:
```bash
python main.py "https://www.youtube.com/channel/UCXXXXXXX" en
```

## Output Data Structure

The script generates JSON and JSONL files with the following structure:

```json
{
  "created": "2023-05-15T12:34:56Z",
  "name": "Video Title",
  "value": "Transcript text..."
}
```

Where:
- `created`: Publication date of the video
- `name`: Title of the video
- `value`: Transcript text (or "no transcript available" if none exists)

## Dataset Files

The script creates the following files:

- `dataset/full_dataset.jsonl`: Complete dataset in JSONL format (one JSON object per line)
- `dataset/train.jsonl`: Training set (approximately 80% of data)
- `dataset/val.jsonl`: Validation set (approximately 10% of data)
- `dataset/test.jsonl`: Test set (approximately 10% of data)
- `dataset/transcripts.json`: Complete dataset as a standard JSON file

## Memory Efficiency

The script processes videos one at a time and writes each transcript directly to the JSONL file instead of keeping all transcripts in memory. This approach allows for processing large channels with many videos without memory issues.
