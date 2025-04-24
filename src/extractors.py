"""
Content extraction utilities for Academic AI Assistant.

This module provides functions to extract content from various sources:
- Web pages
- YouTube videos
- PDF documents
- Search engine results
"""

import os
import sys
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, Optional, Union, List, Tuple
import tempfile
from pathlib import Path
import validators
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import json
import urllib.parse

# Import LangChain's document loaders
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Third-party libraries for PDF processing
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Warning: PDF extraction dependencies not installed")
        # Fallback error message
        def extract_pdf_content(pdf_path):
            return "Error: PDF extraction requires pypdf or PyPDF2 to be installed."

async def extract_website_content(url: str) -> str:
    """Extract the main content from a web page.
    
    Args:
        url (str): The URL of the web page to extract content from
        
    Returns:
        str: The extracted content
    """
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                
                # Parse the HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove unnecessary elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract the text content
                text = soup.get_text(separator='\n', strip=True)
                
                # Clean up the text
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)
                
                # Include the page title
                title = soup.title.string if soup.title else "Untitled Page"
                
                result = f"Title: {title}\nURL: {url}\n\n{text}"
                
                return result
    except Exception as e:
        return f"Error extracting content from {url}: {str(e)}"

def extract_pdf_content(source: Union[str, bytes]) -> str:
    """Extract content from a PDF file.
    
    Args:
        source (Union[str, bytes]): Either a file path or PDF bytes
        
    Returns:
        str: The extracted text content
    """
    try:
        # Check if we have required PDF processing libraries
        try:
            from langchain_community.document_loaders import PyPDFLoader
            import tempfile
        except ImportError:
            return "PDF extraction requires PyPDF and LangChain. Please install with: pip install pypdf langchain-community"
        
        # Create a temporary file if source is bytes
        temp_file = None
        file_path = source
        
        if isinstance(source, bytes):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.write(source)
            temp_file.close()
            file_path = temp_file.name
        
        try:
            # Load PDF with PyPDFLoader
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Try to get title from metadata if available
            title = "Untitled PDF"
            
            # Format the content with page indicators
            text_content = ""
            for i, page in enumerate(pages):
                text_content += f"--- Page {i+1} ---\n"
                text_content += page.page_content
                text_content += "\n\n"
            
            # Format output
            result = f"Title: {title}\nPages: {len(pages)}\n\n{text_content}"
            return result
            
        finally:
            # Clean up temporary file if created
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
                
    except Exception as e:
        return f"Error extracting PDF content: {str(e)}"

def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from a URL.
    
    Args:
        url (str): The YouTube URL
        
    Returns:
        str: The YouTube video ID
    """
    # Try to parse the URL
    parsed_url = urlparse(url)
    
    # Check for youtube.com format
    if 'youtube.com' in parsed_url.netloc:
        if 'watch' in parsed_url.path:
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]
    
    # Check for youtu.be format
    elif 'youtu.be' in parsed_url.netloc:
        # The ID is in the path for youtu.be URLs
        path_parts = parsed_url.path.split('/')
        if len(path_parts) > 1:
            return path_parts[1]
    
    # Try to extract using regex as a fallback
    video_id_match = re.search(r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})', url)
    if video_id_match:
        return video_id_match.group(1)
        
    # If all methods fail, return None or raise an exception
    raise ValueError(f"Could not extract video ID from URL: {url}")

async def extract_youtube_content(url: str) -> str:
    """Extract the transcript from a YouTube video.
    
    Args:
        url (str): The YouTube URL
        
    Returns:
        str: The extracted transcript
    """
    try:
        # Load transcript using LangChain's YoutubeLoader
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()
        
        if not docs:
            return "No transcript content could be extracted from this video."
        
        # Combine all documents into one transcript
        transcript = ""
        for doc in docs:
            transcript += doc.page_content + "\n\n"
        
        return transcript
    except Exception as e:
        return f"Error extracting YouTube transcript: {str(e)}"

async def web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Perform a web search and return the top results.
    
    This function simulates a web search using a search API or service.
    In a production environment, this would connect to a real search API.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        
    Returns:
        List[Dict[str, str]]: List of search results with title, url, and snippet
    """
    try:
        # Use SerpAPI or similar in production
        # For now, we'll use a simple Google search scraper as fallback
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # This is not reliable for production use, just for demo purposes
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers) as response:
                html = await response.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                search_results = []
                
                # Parse Google search results (simplified)
                for result in soup.select('div.g')[:num_results]:
                    title_elem = result.select_one('h3')
                    link_elem = result.select_one('a')
                    snippet_elem = result.select_one('div.VwiC3b')
                    
                    if title_elem and link_elem and snippet_elem:
                        title = title_elem.get_text()
                        url = link_elem['href'] if link_elem.has_attr('href') else ""
                        # Clean URL from Google's redirect
                        if url.startswith('/url?q='):
                            url = url.split('/url?q=')[1].split('&')[0]
                        snippet = snippet_elem.get_text()
                        
                        if url and not url.startswith('/'):
                            search_results.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet
                            })
                
                return search_results
    except Exception as e:
        print(f"Web search error: {str(e)}")
        # Return fallback results in case of error
        return [
            {
                'title': f"Result for: {query}",
                'url': "https://example.com",
                'snippet': "Search result unavailable. Please try again later."
            }
        ]

async def youtube_search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Search for YouTube videos on a topic.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        
    Returns:
        List[Dict[str, str]]: List of YouTube video results with title, url, and description
    """
    try:
        # In production, use YouTube Data API
        # For now, we'll use a simple scraper as fallback
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers) as response:
                html = await response.text()
                
                # Parse YouTube search results
                # This is a simplified version that may break if YouTube changes their HTML structure
                video_ids = re.findall(r"watch\?v=(\S{11})", html)
                unique_ids = []
                for vid in video_ids:
                    if vid not in unique_ids:
                        unique_ids.append(vid)
                
                youtube_results = []
                for video_id in unique_ids[:num_results]:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    # Get video metadata
                    async with session.get(video_url, headers=headers) as video_response:
                        video_html = await video_response.text()
                        title_match = re.search(r'<title>(.*?)</title>', video_html)
                        title = title_match.group(1).replace(' - YouTube', '') if title_match else f"Video {video_id}"
                        
                        youtube_results.append({
                            'title': title,
                            'url': video_url,
                            'id': video_id
                        })
                
                return youtube_results
    except Exception as e:
        print(f"YouTube search error: {str(e)}")
        # Return fallback results in case of error
        return [
            {
                'title': f"YouTube result for: {query}",
                'url': "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                'id': "dQw4w9WgXcQ"
            }
        ]

async def research_topic(topic: str, search_depth: str = "ordinary") -> Dict[str, Any]:
    """Research a topic by performing web searches and collecting information.
    
    Args:
        topic (str): The topic to research
        search_depth (str): "ordinary" or "deep" search
        
    Returns:
        Dict[str, Any]: Collected research information
    """
    result = {
        'topic': topic,
        'web_sources': [],
        'youtube_sources': [],
        'combined_content': "",
        'search_depth': search_depth
    }
    
    # Determine number of results based on search depth
    web_results_count = 3 if search_depth == "ordinary" else 7
    youtube_results_count = 1 if search_depth == "ordinary" else 3
    
    # Run searches in parallel
    web_search_task = asyncio.create_task(web_search(topic, web_results_count))
    youtube_search_task = asyncio.create_task(youtube_search(topic, youtube_results_count))
    
    web_results = await web_search_task
    youtube_results = await youtube_search_task
    
    # Collect web content
    for web_result in web_results:
        try:
            content = await extract_website_content(web_result['url'])
            web_result['content'] = content
            result['web_sources'].append(web_result)
        except Exception as e:
            print(f"Error extracting web content from {web_result['url']}: {str(e)}")
    
    # Collect YouTube content
    for yt_result in youtube_results:
        try:
            content = await extract_youtube_content(yt_result['url'])
            yt_result['content'] = content
            result['youtube_sources'].append(yt_result)
        except Exception as e:
            print(f"Error extracting YouTube content from {yt_result['url']}: {str(e)}")
    
    # Combine all content for processing
    combined_content = f"# Research on: {topic}\n\n"
    
    # Add web sources
    combined_content += "## Web Sources\n\n"
    for i, source in enumerate(result['web_sources']):
        combined_content += f"### Source {i+1}: {source['title']}\n"
        combined_content += f"URL: {source['url']}\n\n"
        combined_content += f"Summary: {source['snippet']}\n\n"
        combined_content += f"Content excerpt: {source['content'][:500]}...\n\n"
    
    # Add YouTube sources
    combined_content += "## YouTube Sources\n\n"
    for i, source in enumerate(result['youtube_sources']):
        combined_content += f"### YouTube Video {i+1}: {source['title']}\n"
        combined_content += f"URL: {source['url']}\n\n"
        transcript_preview = source['content'][:500] + "..." if len(source['content']) > 500 else source['content']
        combined_content += f"Transcript excerpt: {transcript_preview}\n\n"
    
    result['combined_content'] = combined_content
    return result 