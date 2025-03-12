"""
Content extraction utilities for different sources.

This module provides functions to extract content from:
- Web pages
- PDF files
- YouTube videos
"""

import os
import re
import aiohttp
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse, parse_qs
import tempfile
import pytube
from youtube_transcript_api import YouTubeTranscriptApi

# Third-party libraries for PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        # Fallback error message
        def extract_pdf_content(pdf_path):
            return "Error: PDF extraction requires pypdf or PyPDF2 to be installed."

async def extract_website_content(url):
    """
    Extract content from a website URL.
    
    Args:
        url (str): The URL of the webpage to extract content from
        
    Returns:
        str: The extracted text content or an error message
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }) as response:
                if response.status != 200:
                    return f"Error: Failed to fetch content from {url}, status code: {response.status}"
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.extract()
                
                # Get text content
                text = soup.get_text(separator='\n')
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Add title and URL as metadata
                title = soup.title.string if soup.title else "Untitled"
                metadata = f"Title: {title}\nURL: {url}\n\n"
                
                return metadata + text
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"

def extract_pdf_content(pdf_path_or_bytes):
    """
    Extract content from a PDF file.
    
    Args:
        pdf_path_or_bytes: Either a file path to the PDF or the bytes of the PDF
        
    Returns:
        str: The extracted text content or an error message
    """
    try:
        # Check if input is bytes or path
        if isinstance(pdf_path_or_bytes, bytes):
            # Create temporary file to save the bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_path_or_bytes)
                pdf_path = temp_file.name
        else:
            pdf_path = pdf_path_or_bytes
            if not os.path.exists(pdf_path):
                return f"Error: PDF file not found at {pdf_path}"
        
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            metadata = f"Document: {os.path.basename(pdf_path)}\n"
            metadata += f"Pages: {len(pdf_reader.pages)}\n\n"
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"--- Page {page_num + 1} ---\n"
                text += page.extract_text() + "\n\n"
        
        # Clean up temporary file if we created one
        if isinstance(pdf_path_or_bytes, bytes) and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        
        return metadata + text
    except Exception as e:
        # Clean up temporary file if exception occurs
        if isinstance(pdf_path_or_bytes, bytes) and 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return f"Error extracting content from PDF: {str(e)}"

def extract_youtube_id(url):
    """
    Extract YouTube video ID from various URL formats.
    
    Args:
        url (str): YouTube URL in any format
        
    Returns:
        str or None: The YouTube video ID or None if not found
    """
    # Standard YouTube URL format
    parsed_url = urlparse(url)
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            if 'v' in query:
                return query['v'][0]
    
    # Shortened YouTube URL format
    elif parsed_url.netloc == 'youtu.be':
        return parsed_url.path.lstrip('/')
    
    # Try to extract from URL using regex as a fallback
    video_id_match = re.search(r'(?:v=|youtu\.be/|embed/)([^&\?]+)', url)
    if video_id_match:
        return video_id_match.group(1)
    
    return None

async def extract_youtube_content(url):
    """
    Extract content from a YouTube video URL.
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: The extracted content (metadata + transcript) or an error message
    """
    try:
        video_id = extract_youtube_id(url)
        if not video_id:
            return f"Error: Could not extract YouTube video ID from {url}"
        
        # Get video title and metadata using pytube
        yt = pytube.YouTube(url)
        title = yt.title
        channel = yt.author
        description = yt.description
        duration = yt.length  # duration in seconds
        
        # Format duration in minutes and seconds
        minutes, seconds = divmod(duration, 60)
        duration_str = f"{minutes}:{seconds:02d}"
        
        # Get transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Process transcript with timestamps
            transcript_text = ""
            for part in transcript_list:
                # Convert seconds to MM:SS format
                start = part["start"]
                minutes, seconds = divmod(int(start), 60)
                timestamp = f"[{minutes}:{seconds:02d}]"
                
                transcript_text += f"{timestamp} {part['text']}\n"
        except Exception as e:
            transcript_text = f"No transcript available: {str(e)}"
        
        # Combine metadata and transcript
        content = f"Title: {title}\n"
        content += f"Channel: {channel}\n"
        content += f"Duration: {duration_str}\n"
        content += f"URL: {url}\n\n"
        content += f"Description:\n{description}\n\n"
        content += f"Transcript:\n{transcript_text}"
        
        return content
    except Exception as e:
        return f"Error extracting content from YouTube video: {str(e)}" 