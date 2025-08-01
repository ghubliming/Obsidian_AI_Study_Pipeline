"""
Vault parser module for reading and indexing Obsidian markdown files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import frontmatter
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObsidianNote:
    """Represents a single Obsidian note with metadata and content."""
    
    file_path: str
    title: str
    content: str
    raw_content: str
    metadata: Dict = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    math_blocks: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_time: Optional[str] = None
    modified_time: Optional[str] = None

class VaultParser:
    """Parser for Obsidian vaults to extract notes, images, and metadata."""
    
    def __init__(self, vault_path: str):
        """
        Initialize vault parser.
        
        Args:
            vault_path: Path to the Obsidian vault directory
        """
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")
        
        logger.info(f"Initialized VaultParser for: {vault_path}")
    
    def parse_vault(self) -> List[ObsidianNote]:
        """
        Parse the entire vault and return a list of ObsidianNote objects.
        
        Returns:
            List of parsed ObsidianNote objects
        """
        notes = []
        markdown_files = list(self.vault_path.rglob("*.md"))
        
        logger.info(f"Found {len(markdown_files)} markdown files in vault")
        
        for md_file in markdown_files:
            try:
                note = self._parse_single_note(md_file)
                if note:
                    notes.append(note)
            except Exception as e:
                logger.error(f"Error parsing {md_file}: {e}")
        
        logger.info(f"Successfully parsed {len(notes)} notes")
        return notes
    
    def _parse_single_note(self, file_path: Path) -> Optional[ObsidianNote]:
        """
        Parse a single markdown file into an ObsidianNote.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            ObsidianNote object or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Could not read file {file_path}: {e}")
            return None
        
        # Parse frontmatter if present
        try:
            post = frontmatter.loads(content)
            metadata = post.metadata
            main_content = post.content
        except:
            # If no frontmatter, treat entire content as main content
            metadata = {}
            main_content = content
        
        # Extract title (first h1 header or filename)
        title = self._extract_title(main_content, file_path.stem)
        
        # Extract various content types
        images = self._extract_images(main_content)
        math_blocks = self._extract_math_blocks(main_content)
        links = self._extract_links(main_content)
        tags = self._extract_tags(main_content, metadata)
        
        # Clean content for processing (remove frontmatter, normalize)
        clean_content = self._clean_content(main_content)
        
        # Get file timestamps
        stat = file_path.stat()
        
        return ObsidianNote(
            file_path=str(file_path.relative_to(self.vault_path)),
            title=title,
            content=clean_content,
            raw_content=content,
            metadata=metadata,
            images=images,
            math_blocks=math_blocks,
            links=links,
            tags=tags,
            created_time=str(stat.st_ctime),
            modified_time=str(stat.st_mtime)
        )
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from content or use filename as fallback."""
        # Look for first H1 heading
        h1_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        return fallback
    
    def _extract_images(self, content: str) -> List[str]:
        """Extract image references from markdown content."""
        # Obsidian image syntax: ![[image.png]] or ![alt](image.png)
        obsidian_images = re.findall(r'!\[\[([^\]]+\.(png|jpg|jpeg|gif|svg|webp))\]\]', content, re.IGNORECASE)
        markdown_images = re.findall(r'!\[.*?\]\(([^)]+\.(png|jpg|jpeg|gif|svg|webp))\)', content, re.IGNORECASE)
        
        images = [img[0] for img in obsidian_images] + [img[0] for img in markdown_images]
        return list(set(images))  # Remove duplicates
    
    def _extract_math_blocks(self, content: str) -> List[str]:
        """Extract LaTeX math blocks from content."""
        # Block math: $$...$$
        block_math = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)
        # Inline math: $...$
        inline_math = re.findall(r'(?<!\$)\$([^$\n]+)\$(?!\$)', content)
        
        return block_math + inline_math
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract internal and external links."""
        # Obsidian internal links: [[Note Name]]
        internal_links = re.findall(r'\[\[([^\]]+)\]\]', content)
        # Markdown links: [text](url)
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        all_links = internal_links + [link[1] for link in markdown_links]
        return list(set(all_links))
    
    def _extract_tags(self, content: str, metadata: Dict) -> List[str]:
        """Extract tags from content and metadata."""
        tags = set()
        
        # Tags from metadata
        if 'tags' in metadata:
            if isinstance(metadata['tags'], list):
                tags.update(metadata['tags'])
            elif isinstance(metadata['tags'], str):
                tags.update(metadata['tags'].split())
        
        # Inline tags in content: #tag
        inline_tags = re.findall(r'#([a-zA-Z0-9_/-]+)', content)
        tags.update(inline_tags)
        
        return list(tags)
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for processing."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Normalize Obsidian-specific syntax for better processing
        # Convert [[Note Name]] to "Note Name" for better semantic understanding
        content = re.sub(r'\[\[([^\]]+)\]\]', r'"\1"', content)
        
        # Convert ![[image.png]] to a descriptive text
        content = re.sub(r'!\[\[([^\]]+)\]\]', r'[Image: \1]', content)
        
        return content.strip()
    
    def get_vault_stats(self, notes: List[ObsidianNote]) -> Dict:
        """Get statistics about the parsed vault."""
        if not notes:
            return {}
        
        total_images = sum(len(note.images) for note in notes)
        total_math_blocks = sum(len(note.math_blocks) for note in notes)
        total_links = sum(len(note.links) for note in notes)
        all_tags = set()
        for note in notes:
            all_tags.update(note.tags)
        
        return {
            "total_notes": len(notes),
            "total_images": total_images,
            "total_math_blocks": total_math_blocks,
            "total_links": total_links,
            "unique_tags": len(all_tags),
            "all_tags": list(all_tags),
            "average_content_length": sum(len(note.content) for note in notes) / len(notes)
        }