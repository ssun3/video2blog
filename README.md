# video2blog

Convert YouTube videos to comprehensive blog posts using Google Gemini AI.

## Setup

1. Create virtual environment (if not exists):
```bash
# Using uv (recommended)
uv venv

# Or using standard Python
python -m venv .venv
```

2. Activate virtual environment:
```bash
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set API key:
```bash
export GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Force video upload (more accurate):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --multimodal
```

## Output

Creates timestamped directory: `outputs/YYYYMMDD_HHMMSS/VIDEO_ID_Title/`
- `blogpost.html` - Final output (opens automatically)
- `blogpost.md` - Markdown version
- `screenshots/` - Extracted frames at timestamps
- `video.mp4` - Downloaded video

## Notes

- Videos must be <1 hour
- Longer videos use iterative processing for full coverage
- Default mode uses transcripts if available (faster)
- `--multimodal` forces video upload to Gemini (better for visual content)