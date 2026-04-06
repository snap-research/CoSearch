#!/usr/bin/env python3
"""
Search API interaction module

This module handles all interactions with the dense retrieval search API.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any


async def call_search_api(
    query: str,
    search_api_url: str,
    top_k: int = 50,
    semaphore: asyncio.Semaphore = None,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Call dense retrieval API to retrieve documents
    
    API Endpoint: POST /retrieve
    Request format: {"queries": [str], "topk": int, "return_scores": bool}
    Response format: {"result": [[{"document": {...}, "score": float}, ...]]}
    
    Args:
        query: Search query string
        search_api_url: Base URL of search API (e.g., http://gpu021:8000)
        top_k: Number of documents to retrieve
        semaphore: Semaphore for rate limiting
        timeout: Request timeout in seconds
    
    Returns:
        {
            "status": "success" | "error",
            "documents": List[Dict],  # [{"id": ..., "contents": ..., "score": ...}, ...]
            "error": str | None
        }
    """
    # Use context manager for semaphore if provided
    if semaphore:
        async with semaphore:
            return await _do_search_api_call(query, search_api_url, top_k, timeout)
    else:
        return await _do_search_api_call(query, search_api_url, top_k, timeout)


async def _do_search_api_call(
    query: str,
    search_api_url: str,
    top_k: int,
    timeout: float
) -> Dict[str, Any]:
    """Internal function to make the actual search API call"""
    try:
        # Prepare payload according to your API spec
        payload = {
            "queries": [query],      # Single query in a list
            "topk": top_k,
            "return_scores": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                search_api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    result = data.get("result", [])
                    
                    if not result:
                        return {
                            "status": "success",
                            "documents": [],
                            "error": None
                        }
                    
                    # Extract first query's results
                    raw_candidates = result[0]
                    
                    # Parse documents according to your API format
                    # Each item: {"document": {"id": ..., "contents": ..., ...}, "score": ...}
                    documents = []
                    for idx, item in enumerate(raw_candidates):
                        doc_data = item.get("document", {})
                        doc_id = str(doc_data.get("id", ""))
                        
                        # Try different field names for text content
                        contents = (
                            doc_data.get("contents") or 
                            doc_data.get("text") or 
                            doc_data.get("passage") or
                            ""
                        )
                        
                        documents.append({
                            "id": doc_id,
                            "contents": contents,
                            "score": float(item.get("score", 0.0)),
                            "title": doc_data.get("title", "")  # Optional
                        })
                    
                    return {
                        "status": "success",
                        "documents": documents,
                        "error": None
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "error",
                        "documents": [],
                        "error": f"API returned status {response.status}: {error_text}"
                    }
    
    except Exception as e:
        return {
            "status": "error",
            "documents": [],
            "error": f"Search API call failed: {str(e)}"
        }


def format_tool_response(documents: List[Dict], max_doc_length: int = 2000) -> str:
    """
    Format retrieved documents into tool response string
    
    Args:
        documents: List of document dicts with id, contents, score
        max_doc_length: Maximum characters per document (to control context length)
    
    Returns:
        Formatted string for tool response
    """
    if not documents:
        return "No documents found."
    
    lines = []
    for i, doc in enumerate(documents, 1):
        # Get content (try multiple field names)
        contents = (
            doc.get("contents") or 
            doc.get("text") or 
            doc.get("passage") or
            ""
        )
        
        # Truncate if too long
        if len(contents) > max_doc_length:
            contents = contents[:max_doc_length] + "..."
        
        title = doc.get("title", "")
        
        # Format document
        if title:
            doc_str = f"[{i}] Title: {title}\n{contents}"
        else:
            doc_str = f"[{i}] {contents}"
        lines.append(doc_str)
    
    return "\n".join(lines)

def format_tool_response_with_docid_map(documents: List[Dict], max_doc_length: int = 2000) -> str:
    """
    Format retrieved documents into tool response string
    
    Args:
        documents: List of document dicts with id, contents, score
        max_doc_length: Maximum characters per document (to control context length)
    
    Returns:
        Formatted string for tool response
        docid_map: Mapping from index to doc content
    """
    lines = []
    docid_map = {}
    for i, doc in enumerate(documents, 1):
        # Get content (try multiple field names)
        contents = (
            doc.get("contents") or 
            doc.get("text") or 
            doc.get("passage") or
            ""
        )
        
        # Truncate if too long
        if len(contents) > max_doc_length:
            contents = contents[:max_doc_length] + "..."
        
        title = doc.get("title", "")
        
        # Format document
        if title:
            doc_str = f"[{i}] Title: {title}\n{contents}"
        else:
            doc_str = f"[{i}] {contents}"
        lines.append(doc_str)
        docid_map[int(i)] = doc
    
    return "\n".join(lines), docid_map