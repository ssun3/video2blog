# video2blog

Convert YouTube videos to comprehensive blog posts using Google Gemini AI.

## Features

- 🎥 **Video Processing**: Download and analyze YouTube videos
- 📝 **Transcript Support**: Use YouTube transcripts for faster processing
- 🖼️ **Automatic Screenshots**: Extract screenshots at referenced timestamps
- 🔄 **Iterative Processing**: Handle long videos with comprehensive coverage
- 🎨 **Rich Output**: Generate beautiful HTML blog posts with embedded images
- ⚡ **Gemini 2.5 Pro**: Leverages the latest AI model for better analysis
- 🚀 **Python 3.12**: Uses latest Python features including pattern matching

## Requirements

- Python 3.12+
- Google Gemini API key
- uv (recommended) or pip

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/ssun3/video2blog.git
cd video2blog
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Using pip

```bash
git clone https://github.com/ssun3/video2blog.git
cd video2blog
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Force Multimodal Processing

For videos with important visual content:

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --multimodal
```

## Output

Creates timestamped directory: `outputs/YYYYMMDD_HHMMSS/VIDEO_ID_Title/`
- `blogpost.html` - Final output (opens automatically)
- `blogpost.md` - Markdown version
- `screenshots/` - Extracted frames at timestamps
- `video.mp4` - Downloaded video
- `metadata.json` - Video metadata

## Processing Modes

### Transcript Mode (Default)
- Uses YouTube's auto-generated transcripts
- Faster processing
- Good for videos with clear speech

### Multimodal Mode
- Uploads entire video to Gemini
- Better for visual content, code demos, presentations
- Used when no transcript available or with `--multimodal` flag

## Notes

- Videos longer than 30 minutes use iterative processing for complete coverage
- Default timeout is 2 hours (adjustable in script)
- Screenshots are automatically extracted at referenced timestamps

## License

MIT License