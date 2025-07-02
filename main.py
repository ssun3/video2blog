import os
import sys
import re
import json
import shutil
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import yt_dlp
import cv2
import markdown
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_video_info(url):
    """Get video title and other metadata."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title', 'Untitled'),
            'duration': info.get('duration', 0),
            'channel': info.get('channel', 'Unknown'),
            'upload_date': info.get('upload_date', ''),
        }

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    # Replace spaces with underscores and limit length
    filename = filename.replace(' ', '_')[:100]
    return filename

def download_video(url, output_path):
    """Download YouTube video."""
    print(f"Downloading video from {url}...")
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"Video downloaded to {output_path}")
    return output_path

def extract_frame_at_timestamp(video_path, timestamp_seconds, output_path):
    """Extract a frame from video at specific timestamp."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
    cap.release()
    return success

def format_timestamp_for_filename(seconds):
    """Convert seconds to 5-digit string format like 00060."""
    return f"{int(seconds):05d}"

def create_youtube_timestamp_url(video_id, seconds):
    """Create YouTube URL with timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"

def process_gemini_response(response_text, video_id, video_path, screenshots_dir):
    """Process Gemini response to extract timestamps and add screenshots."""
    # Pattern to find timestamps in format [MM:SS] or [HH:MM:SS]
    timestamp_pattern = r'\[(\d{1,2}:\d{2}|\d{1,2}:\d{2}:\d{2})\]'
    
    # Convert response to lines for processing
    lines = response_text.split('\n')
    processed_lines = []
    extracted_screenshots = []
    
    for line in lines:
        # Find all timestamps in the line
        matches = list(re.finditer(timestamp_pattern, line))
        
        if matches:
            # Process line with timestamps
            new_line = line
            screenshots_to_add = []
            
            for match in matches:
                # Extract and convert timestamp to seconds
                timestamp_str = match.group(1)
                parts = timestamp_str.split(':')
                
                try:
                    if len(parts) == 2:
                        seconds = int(parts[0]) * 60 + int(parts[1])
                    elif len(parts) == 3:
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    else:
                        continue
                    
                    # Extract frame
                    filename = f"{format_timestamp_for_filename(seconds)}.jpg"
                    screenshot_path = os.path.join(screenshots_dir, filename)
                    
                    if extract_frame_at_timestamp(video_path, seconds, screenshot_path):
                        # Replace timestamp with link
                        youtube_url = create_youtube_timestamp_url(video_id, seconds)
                        new_line = new_line.replace(match.group(0), f'[{timestamp_str}]({youtube_url})')
                        
                        # Store screenshot info for later addition
                        screenshots_to_add.append((filename, timestamp_str, youtube_url))
                        extracted_screenshots.append(seconds)
                        
                except ValueError:
                    # Skip invalid timestamps
                    continue
            
            # Add the modified line
            processed_lines.append(new_line)
            
            # Add screenshots after the line (if not a header)
            if screenshots_to_add and not line.strip().startswith('#'):
                for filename, timestamp_str, youtube_url in screenshots_to_add:
                    processed_lines.append(f"\n![Screenshot at {timestamp_str}](screenshots/{filename})")
                    processed_lines.append(f"[Link to video]({youtube_url})\n")
        else:
            # No timestamps in this line
            processed_lines.append(line)
    
    return '\n'.join(processed_lines), extracted_screenshots

def upload_video_to_gemini(video_path, display_name=None):
    """Upload video file to Gemini API and wait for processing."""
    import time
    
    print(f"Uploading video file to Gemini: {video_path}")
    
    # Upload the video file
    video_file = genai.upload_file(
        path=video_path, 
        display_name=display_name,
        resumable=True  # Important for large files
    )
    
    print(f"Uploaded file URI: {video_file.uri}")
    
    # Wait for the video to be processed
    while video_file.state.name == "PROCESSING":
        print("Processing video...", end="\r")
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    print("\nVideo processing complete!")
    return video_file

def get_or_upload_video(video_path, display_name):
    """Check if video is already uploaded, otherwise upload it."""
    try:
        # List existing files
        file_list = genai.list_files(page_size=100)
        
        # Check if file with display_name already exists
        for f in file_list:
            if f.display_name == display_name and f.state.name == "ACTIVE":
                print(f"Using existing uploaded file: {f.display_name}")
                return f
    except Exception as e:
        print(f"Could not list files: {e}")
    
    # Upload new file
    return upload_video_to_gemini(video_path, display_name)

def get_transcript(video_id):
    """Get transcript with timestamps from YouTube."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript with timestamps
        formatted_transcript = []
        for entry in transcript:
            start_time = entry['start']
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_transcript.append(f"{timestamp} {entry['text']}")
        
        return "\n".join(formatted_transcript)
    except Exception as e:
        print(f"Warning: Could not get transcript: {e}")
        return None

def generate_comprehensive_blog_post(model, prompt_content, video_info, video_output_dir, video_id, video_path, screenshots_dir, use_transcript_only=False):
    """Generate a comprehensive blog post with iterative prompting for long videos."""
    all_responses = []
    all_extracted_screenshots = []
    video_duration_minutes = video_info['duration'] // 60
    
    # Initial prompt
    initial_prompt = f"""Please analyze this video and create a detailed blog post about it. 

This video is {video_duration_minutes} minutes long. Start from the beginning and work through the video chronologically.

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

Be VERY CAREFUL to check what timestamp you've actually reached. The video is {video_duration_minutes} minutes long, so if you haven't reached close to [{video_duration_minutes}:00], you should use CONTINUE, not COMPLETE."""

    print("\n📝 Generating initial blog post...")
    try:
        if use_transcript_only:
            response = model.generate_content(
                prompt_content,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 8192,
                },
                request_options={"timeout": 900}  # 15 minutes timeout
            )
        else:
            response = model.generate_content(
                [prompt_content, initial_prompt],
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 8192,
                },
                request_options={"timeout": 900}  # 15 minutes timeout
            )
        all_responses.append(response.text)
        print("✅ Initial response generated")
    except Exception as e:
        print(f"Error generating initial content: {e}")
        return None, []
    
    # Check if we need to continue based on the response
    max_iterations = 10  # Safety limit to prevent infinite loops
    iteration = 1
    
    while iteration < max_iterations:
        last_response = all_responses[-1]
        
        # Check if the model indicated it's complete or needs to continue
        should_continue = False
        if "COMPLETE:" in last_response or "I have analyzed the entire video" in last_response:
            # Double-check by finding the last timestamp
            import re
            all_timestamps = re.findall(r'\[(\d+):(\d+)\]', last_response)
            if all_timestamps:
                max_minute = max(int(m) for m, s in all_timestamps)
                if max_minute < video_duration_minutes - 5:  # If more than 5 minutes missing
                    print(f"⚠️ Model said COMPLETE but only reached [{max_minute}:xx] out of {video_duration_minutes} minutes")
                    print("📝 Forcing continuation...")
                    should_continue = True
                else:
                    print(f"✅ Model indicates full video coverage achieved (reached [{max_minute}:xx])")
                    break
            else:
                print("✅ Model indicates full video coverage achieved")
                break
        else:
            should_continue = True
        
        if should_continue or "CONTINUE:" in last_response or iteration == 1:
            # Extract where we are in the video from the response
            import re
            timestamp_match = re.search(r'covered up to \[(\d+:\d+)\]', last_response)
            current_position = timestamp_match.group(1) if timestamp_match else "unknown"
            
            # Also check actual progress
            all_timestamps = re.findall(r'\[(\d+):(\d+)\]', last_response)
            if all_timestamps:
                max_minute = max(int(m) for m, s in all_timestamps)
                actual_position = f"[{max_minute}:xx]"
                print(f"\n📝 Generating continuation {iteration} (model says: {current_position}, actual: {actual_position})...")
            else:
                print(f"\n📝 Generating continuation {iteration} (currently at {current_position})...")
            
            continuation_prompt = f"""Please continue analyzing the video from where you left off.

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

IMPORTANT: The video is {video_duration_minutes} minutes long. You should ONLY say COMPLETE if you've reached timestamps close to [{video_duration_minutes}:00]. If your last timestamp is significantly before that, use CONTINUE."""
            
            try:
                if use_transcript_only:
                    # Include context from last response
                    context = all_responses[-1][-2000:]  # Last ~2000 chars for context
                    full_prompt = f"Previous section ended with:\n...{context}\n\n{continuation_prompt}"
                    response = model.generate_content(
                        full_prompt,
                        generation_config={
                            'temperature': 0.7,
                            'max_output_tokens': 8192,
                        },
                        request_options={"timeout": 900}
                    )
                else:
                    response = model.generate_content(
                        [prompt_content, continuation_prompt],
                        generation_config={
                            'temperature': 0.7,
                            'max_output_tokens': 8192,
                        },
                        request_options={"timeout": 900}
                    )
                all_responses.append(response.text)
                print(f"✅ Continuation {iteration} generated")
                iteration += 1
            except Exception as e:
                print(f"Error generating continuation {iteration}: {e}")
                break
        else:
            # No clear signal, but let's check if we have enough coverage
            print("⚠️ No clear continuation signal found, checking coverage...")
            break
    
    if iteration >= max_iterations:
        print(f"⚠️ Reached maximum iterations ({max_iterations}), stopping...")
    
    # Combine all responses
    combined_response = "\n\n---\n\n".join(all_responses)
    
    # Save raw combined response
    raw_response_path = os.path.join(video_output_dir, "raw_response_combined.md")
    with open(raw_response_path, "w") as f:
        f.write(combined_response)
    
    # Process each response for screenshots
    processed_parts = []
    for i, response_text in enumerate(all_responses):
        print(f"\n🖼️  Processing screenshots for part {i+1}...")
        processed_content, screenshot_timestamps = process_gemini_response(
            response_text, video_id, video_path, screenshots_dir
        )
        processed_parts.append(processed_content)
        all_extracted_screenshots.extend(screenshot_timestamps)
    
    # Combine processed content
    final_content = "\n\n---\n\n".join(processed_parts)
    
    return final_content, all_extracted_screenshots

def main():
    # Get the video URL from command-line arguments
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
    print(f"Video Title: {video_info['title']}")
    print(f"Channel: {video_info['channel']}")
    
    # Create output directory structure with date-timestamp
    base_output_dir = "outputs"
    date_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir_name = f"{video_id}_{sanitize_filename(video_info['title'])}"
    video_output_dir = os.path.join(base_output_dir, date_timestamp, video_dir_name)
    screenshots_dir = os.path.join(video_output_dir, "screenshots")
    
    # Create directories
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Save video metadata
    metadata_path = os.path.join(video_output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            'video_id': video_id,
            'url': video_url,
            'title': video_info['title'],
            'channel': video_info['channel'],
            'duration': video_info['duration'],
            'upload_date': video_info['upload_date'],
            'processed_date': datetime.now().isoformat(),
        }, f, indent=2)
    
    # Get transcript
    print("Fetching transcript...")
    transcript = get_transcript(video_id)
    
    # Download video
    video_path = os.path.join(video_output_dir, "video.mp4")
    if not os.path.exists(video_path):
        download_video(video_url, video_path)
    else:
        print(f"Using existing video file: {video_path}")
    
    # Decide whether to use video upload or transcript
    use_transcript_only = False
    video_file = None
    
    if force_multimodal or not transcript or len(transcript) < 1000:
        # Try to upload video
        try:
            if force_multimodal:
                print("🎬 Multimodal mode forced - will upload video to Gemini")
            else:
                print("📹 No/insufficient transcript available, attempting video upload...")
            display_name = f"{video_id}_{sanitize_filename(video_info['title'])}"
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
        # We have a good transcript and multimodal not forced
        print("📝 Using transcript-based processing (faster)")
        use_transcript_only = True
    
    # Enhanced prompt for multimodal processing
    if use_transcript_only:
        prompt_template = f"""Based on the following transcript from the YouTube video "{video_info['title']}" ({video_url}), create a detailed blog post.

TRANSCRIPT:
{transcript[:50000] if transcript else 'No transcript available'}

Please analyze this content and create a detailed blog post about it. 

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

The blog post should be well-structured and include all relevant information from the video."""
    else:
        prompt_template = """Please analyze this video and create a detailed blog post about it. 

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

The blog post should be well-structured and include all relevant information from the video."""

    # Create the model
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Generate comprehensive blog post
    print(f"\n📊 Video duration: {video_info['duration'] // 60}:{video_info['duration'] % 60:02d}")
    if video_info['duration'] > 1800:
        print("⏰ Long video detected - will use iterative prompting for comprehensive coverage")
    
    if use_transcript_only:
        processed_content, screenshot_timestamps = generate_comprehensive_blog_post(
            model, prompt_template, video_info, video_output_dir, 
            video_id, video_path, screenshots_dir, use_transcript_only=True
        )
    else:
        processed_content, screenshot_timestamps = generate_comprehensive_blog_post(
            model, video_file, video_info, video_output_dir, 
            video_id, video_path, screenshots_dir, use_transcript_only=False
        )
    
    if processed_content is None:
        print("Error: Failed to generate blog post")
        sys.exit(1)
    
    # Save processed markdown
    processed_md_path = os.path.join(video_output_dir, "blogpost.md")
    with open(processed_md_path, "w") as f:
        f.write(processed_content)
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{video_info['title']} - Blog Post</title>
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
            <strong>Video:</strong> <a href="{video_url}" target="_blank">{video_info['title']}</a><br>
            <strong>Channel:</strong> {video_info['channel']}<br>
            <strong>Extracted Screenshots:</strong> {len(screenshot_timestamps)}<br>
            <strong>Processed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
{markdown.markdown(processed_content, extensions=['fenced_code', 'tables'])}
    </article>
</body>
</html>
"""
    
    # Save HTML
    html_path = os.path.join(video_output_dir, "blogpost.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Output directory: {video_output_dir}")
    print(f"📄 Blog post: {html_path}")
    print(f"🖼️  Screenshots: {len(screenshot_timestamps)} extracted")
    print(f"\nOpening blog post...")
    
    # Open the HTML file
    os.system(f"open '{html_path}'")

if __name__ == "__main__":
    main()