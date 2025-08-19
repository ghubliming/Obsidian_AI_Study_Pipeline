"""
Preprocessing module for cleaning and preparing vault content for AI processing.
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from bs4 import BeautifulSoup

from ..vault_parser import ObsidianNote

logger = logging.getLogger(__name__)

@dataclass
class ContentChunk:
    """Represents a chunk of processed content with metadata."""
    
    text: str
    source_note: str
    source_path: str
    chunk_index: int
    chunk_type: str  # 'paragraph', 'section', 'list', 'code', 'math'
    metadata: Dict = None

class ContentPreprocessor:
    """Preprocesses Obsidian notes for AI processing."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize content preprocessor.
        
        Args:
            chunk_size: Maximum size of content chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            min_chunk_size: Minimum size for a chunk to be considered valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(f"Initialized ContentPreprocessor with chunk_size={chunk_size}")
    
    def preprocess_notes(self, notes: List[ObsidianNote]) -> List[ContentChunk]:
        """
        Preprocess a list of notes into content chunks.
        
        Args:
            notes: List of ObsidianNote objects
            
        Returns:
            List of ContentChunk objects
        """
        all_chunks = []
        
        for note in notes:
            try:
                chunks = self._process_single_note(note)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing note {note.file_path}: {e}")
        
        logger.info(f"Generated {len(all_chunks)} content chunks from {len(notes)} notes")
        return all_chunks
    
    def _process_single_note(self, note: ObsidianNote) -> List[ContentChunk]:
        """Process a single note into content chunks."""
        chunks = []
        
        # Clean the content
        cleaned_content = self._deep_clean_content(note.content)
        
        # Split into sections first
        sections = self._split_into_sections(cleaned_content)
        
        chunk_index = 0
        for section_type, section_content in sections:
            if len(section_content.strip()) < self.min_chunk_size:
                continue
            
            # Further chunk large sections
            if len(section_content) > self.chunk_size:
                sub_chunks = self._chunk_text(section_content)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk.strip()) >= self.min_chunk_size:
                        chunks.append(ContentChunk(
                            text=sub_chunk.strip(),
                            source_note=note.title,
                            source_path=note.file_path,
                            chunk_index=chunk_index,
                            chunk_type=section_type,
                            metadata={
                                'tags': note.tags,
                                'images': note.images,
                                'has_math': len(note.math_blocks) > 0,
                                'link_count': len(note.links)
                            }
                        ))
                        chunk_index += 1
            else:
                chunks.append(ContentChunk(
                    text=section_content.strip(),
                    source_note=note.title,
                    source_path=note.file_path,
                    chunk_index=chunk_index,
                    chunk_type=section_type,
                    metadata={
                        'tags': note.tags,
                        'images': note.images,
                        'has_math': len(note.math_blocks) > 0,
                        'link_count': len(note.links)
                    }
                ))
                chunk_index += 1
        
        return chunks
    
    def _deep_clean_content(self, content: str) -> str:
        """Perform deep cleaning of content."""
        # Remove HTML tags if any
        content = BeautifulSoup(content, 'html.parser').get_text()
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Clean up markdown formatting while preserving structure
        # Remove excessive markdown formatting
        content = re.sub(r'\*{3,}', '**', content)  # Reduce multiple asterisks
        content = re.sub(r'#{4,}', '###', content)  # Limit heading levels
        
        # Normalize quotes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        content = re.sub(r'\.{3,}', '...', content)
        content = re.sub(r'\?{2,}', '?', content)
        content = re.sub(r'!{2,}', '!', content)
        
        return content.strip()
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content into logical sections."""
        sections = []
        
        # Split by headers first
        header_pattern = r'^(#{1,6})\s+(.+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        if len(parts) == 1:
            # No headers found, split by paragraphs
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    sections.append(('paragraph', para.strip()))
        else:
            # Process header-based sections
            current_content = parts[0].strip()
            if current_content:
                sections.append(('paragraph', current_content))
            
            i = 1
            while i < len(parts):
                if i + 2 < len(parts):
                    header_level = parts[i]
                    header_text = parts[i + 1]
                    section_content = parts[i + 2]
                    
                    full_section = f"{header_level} {header_text}\n{section_content}".strip()
                    sections.append(('section', full_section))
                    i += 3
                else:
                    break
        
        # Further process each section for special content types
        processed_sections = []
        for section_type, section_content in sections:
            # Check for code blocks
            if '```' in section_content:
                processed_sections.append(('code', section_content))
            # Check for math content
            elif '$' in section_content or '\\(' in section_content:
                processed_sections.append(('math', section_content))
            # Check for lists
            elif re.search(r'^[*+-]\s', section_content, re.MULTILINE) or re.search(r'^\d+\.\s', section_content, re.MULTILINE):
                processed_sections.append(('list', section_content))
            else:
                processed_sections.append((section_type, section_content))
        
        return processed_sections
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence ending within reasonable distance
                for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
                else:
                    # Look for word boundaries
                    for i in range(end, max(start + self.chunk_size // 2, end - 50), -1):
                        if text[i].isspace():
                            end = i
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[ContentChunk]) -> Dict:
        """Get statistics about the processed chunks."""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        chunk_types = {}
        notes_count = len(set(chunk.source_note for chunk in chunks))
        
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "unique_source_notes": notes_count,
            "avg_chunk_length": np.mean(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "chunk_types": chunk_types,
            "chunks_with_math": sum(1 for chunk in chunks if chunk.metadata and chunk.metadata.get('has_math', False)),
            "chunks_with_images": sum(1 for chunk in chunks if chunk.metadata and chunk.metadata.get('images'))
        }