#!/usr/bin/env python3
"""Convert YouTube videos to comprehensive blog posts using Google Gemini AI."""

import os
import sys
import re
import json
import time
from datetime import datetime
from pathlib import Path
from functools import partial, lru_cache
from typing import Any, Callable, Iterator, NamedTuple
from collections.abc import Sequence

from dotenv import load_dotenv
import google.generativeai as genai
import yt_dlp
import cv2
import markdown
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configuration
GEMINI_MODEL = "gemini-2.5-pro"
REQUEST_TIMEOUT = 7200  # 2 hours
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.7
MAX_ITERATIONS = 10


class VideoMetadata(NamedTuple):
    """Video metadata."""
    video_id: str
    url: str
    title: str
    channel: str
    duration: int
    upload_date: str


class Timestamp(NamedTuple):
    """Timestamp representation."""
    seconds: int
    formatted: str
    
    @classmethod
    def from_seconds(cls, seconds: int) -> "Timestamp":
        """Create timestamp from seconds."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            formatted = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            formatted = f"{minutes:02d}:{secs:02d}"
        
        return cls(seconds, formatted)
    
    @classmethod
    def from_string(cls, timestamp_str: str) -> "Timestamp":
        """Parse timestamp from string."""
        parts = timestamp_str.split(":")
        
        match len(parts):
            case 2:
                minutes, secs = map(int, parts)
                seconds = minutes * 60 + secs
            case 3:
                hours, minutes, secs = map(int, parts)
                seconds = hours * 3600 + minutes * 60 + secs
            case _:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}")
        
        return cls(seconds, timestamp_str)


# Functional utilities
def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single function."""
    def inner(data: Any) -> Any:
        result = data
        for func in reversed(functions):
            result = func(result)
        return result
    return inner


def pipe(data: Any, *functions: Callable) -> Any:
    """Pipe data through multiple functions."""
    result = data
    for func in functions:
        result = func(result)
    return result


def safe_execute(func: Callable, default: Any = None) -> Callable:
    """Wrap function to return default on exception."""
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Warning: {func.__name__} failed: {e}")
            return default
    return wrapper


# URL processing
def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^?]+)",
    ]
    
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    
    return None


# Video metadata
@lru_cache(maxsize=32)
def get_video_info(url: str) -> VideoMetadata:
    """Get video metadata."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from {url}")
        
        return VideoMetadata(
            video_id=video_id,
            url=url,
            title=info.get("title", "Untitled"),
            channel=info.get("channel", "Unknown"),
            duration=info.get("duration", 0),
            upload_date=info.get("upload_date", ""),
        )


# Filename utilities
def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Remove invalid characters from filename."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join(c for c in filename if c not in invalid_chars)
    return sanitized.replace(" ", "_")[:max_length].rstrip(". ")


# Video download
def download_video(url: str, output_path: Path) -> Path:
    """Download YouTube video."""
    print(f"Downloading video from {url}...")
    
    if output_path.exists():
        print(f"Video already exists at {output_path}")
        return output_path
    
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"Video downloaded to {output_path}")
    return output_path


# Frame extraction
def extract_frame_at_timestamp(video_path: Path, timestamp_seconds: int, output_path: Path) -> bool:
    """Extract a frame from video at specific timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(str(output_path), frame)
    
    cap.release()
    return success


# Transcript processing
def get_transcript(video_id: str) -> str | None:
    """Get transcript with timestamps from YouTube."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript with timestamps
        formatted_lines = [
            f"[{Timestamp.from_seconds(int(entry['start'])).formatted}] {entry['text']}"
            for entry in transcript_list
        ]
        
        return "\n".join(formatted_lines)
    except Exception as e:
        print(f"Warning: Could not get transcript: {e}")
        return None


# Gemini file management
def upload_video_to_gemini(video_path: Path, display_name: str) -> Any:
    """Upload video file to Gemini API and wait for processing."""
    print(f"Uploading video file to Gemini: {video_path}")
    
    video_file = genai.upload_file(
        path=str(video_path),
        display_name=display_name,
        resumable=True,
    )
    
    print(f"Uploaded file URI: {video_file.uri}")
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("Processing video...", end="\r")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    print("\nVideo processing complete!")
    return video_file


def get_or_upload_video(video_path: Path, display_name: str) -> Any:
    """Check if video is already uploaded, otherwise upload it."""
    try:
        # List existing files
        for f in genai.list_files(page_size=100):
            if f.display_name == display_name and f.state.name == "ACTIVE":
                print(f"Using existing uploaded file: {f.display_name}")
                return f
    except Exception as e:
        print(f"Could not list files: {e}")
    
    # Upload new file
    return upload_video_to_gemini(video_path, display_name)


# Timestamp processing
def process_timestamp_match(match: re.Match, video_id: str, video_path: Path, screenshots_dir: Path) -> tuple[str, str | None]:
    """Process a single timestamp match."""
    timestamp_str = match.group(1)
    
    try:
        timestamp = Timestamp.from_string(timestamp_str)
        screenshot_filename = f"{timestamp.seconds:05d}.jpg"
        screenshot_path = screenshots_dir / screenshot_filename
        
        if extract_frame_at_timestamp(video_path, timestamp.seconds, screenshot_path):
            youtube_url = f"https://www.youtube.com/watch?v={video_id}&t={timestamp.seconds}s"
            replacement = f"[{timestamp_str}]({youtube_url})"
            return replacement, (timestamp, screenshot_filename, youtube_url)
        
    except ValueError:
        pass
    
    return match.group(0), None


def process_line_with_timestamps(
    line: str,
    timestamp_pattern: re.Pattern,
    video_id: str,
    video_path: Path,
    screenshots_dir: Path
) -> tuple[str, list[tuple[Timestamp, str, str]]]:
    """Process a single line for timestamps."""
    screenshots_to_add = []
    
    def replacer(match: re.Match) -> str:
        replacement, screenshot_info = process_timestamp_match(
            match, video_id, video_path, screenshots_dir
        )
        if screenshot_info:
            screenshots_to_add.append(screenshot_info)
        return replacement
    
    new_line = timestamp_pattern.sub(replacer, line)
    return new_line, screenshots_to_add


def process_gemini_response(response_text: str, video_id: str, video_path: Path, screenshots_dir: Path) -> tuple[str, list[int]]:
    """Process Gemini response to extract timestamps and add screenshots."""
    timestamp_pattern = re.compile(r'\[(\d{1,2}:\d{2}|\d{1,2}:\d{2}:\d{2})\]')
    lines = response_text.split('\n')
    processed_lines = []
    extracted_screenshots = []
    
    for line in lines:
        # Process line for timestamps
        new_line, screenshots_to_add = process_line_with_timestamps(
            line, timestamp_pattern, video_id, video_path, screenshots_dir
        )
        
        processed_lines.append(new_line)
        
        # Add screenshots after the line (if not a header)
        if screenshots_to_add and not line.strip().startswith('#'):
            for timestamp, filename, youtube_url in screenshots_to_add:
                processed_lines.extend([
                    f"\n![Screenshot at {timestamp.formatted}](screenshots/{filename})",
                    f"[Link to video]({youtube_url})\n"
                ])
                extracted_screenshots.append(timestamp.seconds)
    
    return '\n'.join(processed_lines), extracted_screenshots


# Prompt templates
def get_initial_prompt(duration_minutes: int) -> str:
    """Get initial prompt for video analysis."""
    return f"""Please analyze this video and create a detailed blog post about it. 

This video is {duration_minutes} minutes long. Start from the beginning and work through the video chronologically.

IMPORTANT: Include timestamps in square brackets [MM:SS] or [HH:MM:SS] throughout your response whenever you reference specific moments from the video. For example:
- "At [02:15], the presenter explains..."
- "The code example shown at [05:30] demonstrates..."
- "Starting from [10:45], we see how to..."

Structure your blog post with:
1. A compelling title
2. An introduction
3. Main sections with clear headers
4. Timestamps for important moments, code examples, or visual demonstrations
5. Code snippets with proper formatting
6. A conclusion

Make sure to include timestamps especially for:
- Key concepts being introduced
- Code examples or demonstrations
- Visual diagrams or important screenshots
- Topic transitions

Make your response as long and detailed as possible, covering EVERY aspect of the video without condensing or summarizing. The blog post should be comprehensive and include all relevant information from the video.

IMPORTANT: At the end of your response, you MUST include one of these exact lines:
- "CONTINUE: I have covered up to [MM:SS] of the video. There is more content to analyze."
- "COMPLETE: I have analyzed the entire video up to the end."

Be VERY CAREFUL to check what timestamp you've actually reached. The video is {duration_minutes} minutes long, so if you haven't reached close to [{duration_minutes}:00], you should use CONTINUE, not COMPLETE."""


def get_continuation_prompt(duration_minutes: int) -> str:
    """Get continuation prompt for long videos."""
    return f"""Please continue analyzing the video from where you left off.

Remember to:
- Start from where you previously stopped
- Include timestamps [MM:SS] for ALL content
- Be as detailed and comprehensive as possible
- Cover EVERY topic, example, and demonstration shown
- Include all code snippets, diagrams, and visual elements
- Capture all Q&A or discussion segments

At the end of your response, you MUST include one of these exact lines:
- "CONTINUE: I have covered up to [MM:SS] of the video. There is more content to analyze."
- "COMPLETE: I have analyzed the entire video up to the end."

IMPORTANT: The video is {duration_minutes} minutes long. You should ONLY say COMPLETE if you've reached timestamps close to [{duration_minutes}:00]. If your last timestamp is significantly before that, use CONTINUE."""


def get_transcript_prompt(title: str, url: str, transcript: str) -> str:
    """Get prompt for transcript-based processing."""
    return f"""Based on the following transcript from the YouTube video "{title}" ({url}), create a detailed blog post.

TRANSCRIPT:
{transcript[:50000]}

Please analyze this content and create a detailed blog post about it. 

IMPORTANT: Include timestamps in square brackets [MM:SS] or [HH:MM:SS] throughout your response whenever you reference specific moments from the video.

Structure your blog post with:
1. A compelling title
2. An introduction
3. Main sections with clear headers
4. Timestamps for important moments, code examples, or visual demonstrations
5. Code snippets with proper formatting
6. A conclusion

The blog post should be well-structured and include all relevant information from the video."""


# Response analysis
def is_response_complete(response: str, duration_minutes: int) -> bool:
    """Check if response indicates complete video coverage."""
    if "COMPLETE:" in response or "I have analyzed the entire video" in response:
        # Verify actual coverage
        all_timestamps = re.findall(r'\[(\d+):(\d+)\]', response)
        if all_timestamps:
            max_minute = max(int(m) for m, s in all_timestamps)
            return max_minute >= duration_minutes - 5  # 5-minute tolerance
        return True  # Trust the model if no timestamps found
    return False


# Blog generation
def generate_comprehensive_blog_post(
    model: Any,
    prompt_content: Any,
    video_info: VideoMetadata,
    video_output_dir: Path,
    video_path: Path,
    screenshots_dir: Path,
    use_transcript_only: bool = False
) -> tuple[str, list[int]]:
    """Generate a comprehensive blog post with iterative prompting for long videos."""
    all_responses = []
    all_extracted_screenshots = []
    duration_minutes = video_info.duration // 60
    
    # Initial generation
    print("\n📝 Generating initial blog post...")
    try:
        if use_transcript_only:
            response = model.generate_content(
                prompt_content,
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                request_options={"timeout": REQUEST_TIMEOUT}
            )
        else:
            initial_prompt = get_initial_prompt(duration_minutes)
            response = model.generate_content(
                [prompt_content, initial_prompt],
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                request_options={"timeout": REQUEST_TIMEOUT}
            )
        all_responses.append(response.text)
        print("✅ Initial response generated")
    except Exception as e:
        print(f"Error generating initial content: {e}")
        return None, []
    
    # Continue for long videos
    iteration = 1
    while iteration < MAX_ITERATIONS and not use_transcript_only:
        if is_response_complete(all_responses[-1], duration_minutes):
            print(f"✅ Model indicates full video coverage achieved")
            break
        
        print(f"\n📝 Generating continuation {iteration}...")
        continuation_prompt = get_continuation_prompt(duration_minutes)
        
        try:
            response = model.generate_content(
                [prompt_content, continuation_prompt],
                generation_config={
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                request_options={"timeout": REQUEST_TIMEOUT}
            )
            all_responses.append(response.text)
            print(f"✅ Continuation {iteration} generated")
            iteration += 1
        except Exception as e:
            print(f"Error generating continuation {iteration}: {e}")
            break
    
    # Save raw combined response
    combined_response = "\n\n---\n\n".join(all_responses)
    raw_response_path = video_output_dir / "raw_response_combined.md"
    raw_response_path.write_text(combined_response, encoding="utf-8")
    
    # Process each response for screenshots
    processed_parts = []
    for i, response_text in enumerate(all_responses):
        print(f"\n🖼️  Processing screenshots for part {i+1}...")
        processed_content, screenshot_timestamps = process_gemini_response(
            response_text, video_info.video_id, video_path, screenshots_dir
        )
        processed_parts.append(processed_content)
        all_extracted_screenshots.extend(screenshot_timestamps)
    
    final_content = "\n\n---\n\n".join(processed_parts)
    return final_content, all_extracted_screenshots


# HTML generation
def generate_html(content: str, video_info: VideoMetadata, screenshot_count: int) -> str:
    """Generate HTML from markdown content."""
    html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{video_info.title} - Blog Post</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 854px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        article {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #666;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
        }}
        pre {{
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <article>
        <div class="metadata">
            <strong>Video:</strong> <a href="{video_info.url}" target="_blank">{video_info.title}</a><br>
            <strong>Channel:</strong> {video_info.channel}<br>
            <strong>Extracted Screenshots:</strong> {screenshot_count}<br>
            <strong>Processed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
{html_content}
    </article>
</body>
</html>
"""


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_url> [--multimodal]")
        print("Options:")
        print("  --multimodal    Force multimodal video processing (upload video to Gemini)")
        sys.exit(1)
    
    video_url = sys.argv[1]
    force_multimodal = "--multimodal" in sys.argv
    
    # Extract video ID and get metadata
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Error: Could not extract video ID from URL")
        sys.exit(1)
    
    print(f"Video ID: {video_id}")
    video_info = get_video_info(video_url)
    print(f"Video Title: {video_info.title}")
    print(f"Channel: {video_info.channel}")
    
    # Create output directory structure
    base_output_dir = Path("outputs")
    date_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir_name = f"{video_id}_{sanitize_filename(video_info.title)}"
    video_output_dir = base_output_dir / date_timestamp / video_dir_name
    screenshots_dir = video_output_dir / "screenshots"
    
    # Create directories
    video_output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(exist_ok=True)
    
    # Save video metadata
    metadata_path = video_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps({
        'video_id': video_id,
        'url': video_url,
        'title': video_info.title,
        'channel': video_info.channel,
        'duration': video_info.duration,
        'upload_date': video_info.upload_date,
        'processed_date': datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")
    
    # Get transcript
    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    
    # Download video
    video_path = video_output_dir / "video.mp4"
    download_video(video_url, video_path)
    
    # Create model
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Decide processing mode
    use_transcript_only = False
    video_file = None
    
    if force_multimodal or not transcript or len(transcript) < 1000:
        # Try to upload video
        try:
            if force_multimodal:
                print("🎬 Multimodal mode forced - will upload video to Gemini")
            else:
                print("📹 No/insufficient transcript available, attempting video upload...")
            
            display_name = f"{video_id}_{sanitize_filename(video_info.title)}"
            video_file = get_or_upload_video(video_path, display_name)
            use_transcript_only = False
        except Exception as e:
            print(f"❌ Video upload failed: {e}")
            if transcript and len(transcript) > 500:
                print("📝 Falling back to transcript processing")
                use_transcript_only = True
            else:
                print("❌ Error: No transcript and video upload failed")
                sys.exit(1)
    else:
        print("📝 Using transcript-based processing (faster)")
        use_transcript_only = True
    
    # Generate blog post
    print(f"\n📊 Video duration: {video_info.duration // 60}:{video_info.duration % 60:02d}")
    if video_info.duration > 1800:
        print("⏰ Long video detected - will use iterative prompting for comprehensive coverage")
    
    if use_transcript_only:
        prompt = get_transcript_prompt(video_info.title, video_info.url, transcript)
        processed_content, screenshot_timestamps = generate_comprehensive_blog_post(
            model, prompt, video_info, video_output_dir, 
            video_path, screenshots_dir, use_transcript_only=True
        )
    else:
        processed_content, screenshot_timestamps = generate_comprehensive_blog_post(
            model, video_file, video_info, video_output_dir, 
            video_path, screenshots_dir, use_transcript_only=False
        )
    
    if processed_content is None:
        print("Error: Failed to generate blog post")
        sys.exit(1)
    
    # Save outputs
    processed_md_path = video_output_dir / "blogpost.md"
    processed_md_path.write_text(processed_content, encoding="utf-8")
    
    # Generate and save HTML
    html_content = generate_html(processed_content, video_info, len(screenshot_timestamps))
    html_path = video_output_dir / "blogpost.html"
    html_path.write_text(html_content, encoding="utf-8")
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output directory: {video_output_dir}")
    print(f"📄 Blog post: {html_path}")
    print(f"🖼️  Screenshots: {len(screenshot_timestamps)} extracted")
    print(f"\nOpening blog post...")
    
    # Open the HTML file
    os.system(f"open '{html_path}'")


if __name__ == "__main__":
    main()