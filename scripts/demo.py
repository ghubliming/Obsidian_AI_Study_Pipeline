#!/usr/bin/env python3
"""
Demo script showing vault parsing functionality without heavy dependencies.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from obsidian_ai_study_pipeline.vault_parser import VaultParser

def main():
    print("🚀 Obsidian AI Study Pipeline - Demo")
    print("=" * 50)
    
    # Parse the example vault
    vault_path = Path(__file__).parent / "examples" / "sample_vault"
    
    if not vault_path.exists():
        print(f"❌ Example vault not found at: {vault_path}")
        return 1
    
    print(f"📂 Parsing vault: {vault_path}")
    print()
    
    try:
        # Initialize parser
        parser = VaultParser(str(vault_path))
        
        # Parse vault
        notes = parser.parse_vault()
        
        print(f"✅ Successfully parsed {len(notes)} notes!")
        print()
        
        # Show details for each note
        for i, note in enumerate(notes, 1):
            print(f"{i}. **{note.title}**")
            print(f"   📄 File: {note.file_path}")
            print(f"   📝 Content: {len(note.content)} characters")
            print(f"   🏷️ Tags: {', '.join(note.tags) if note.tags else 'None'}")
            print(f"   🖼️ Images: {len(note.images)}")
            print(f"   🔗 Links: {len(note.links)}")
            print(f"   📊 Math blocks: {len(note.math_blocks)}")
            print(f"   📖 Preview: {note.content[:150]}...")
            print()
        
        # Show vault statistics
        stats = parser.get_vault_stats(notes)
        print("📊 Vault Statistics:")
        print(f"   • Total notes: {stats['total_notes']}")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • Total math blocks: {stats['total_math_blocks']}")
        print(f"   • Total links: {stats['total_links']}")
        print(f"   • Unique tags: {stats['unique_tags']}")
        print(f"   • Average content length: {stats['average_content_length']:.0f} chars")
        print()
        
        # Show all tags found
        if stats['all_tags']:
            print("🏷️ All Tags Found:")
            for tag in sorted(stats['all_tags']):
                print(f"   • #{tag}")
        
        print()
        print("✅ Demo completed successfully!")
        print()
        print("💡 Next steps:")
        print("1. Install full dependencies: pip install -r requirements.txt")
        print("2. Install Ollama for AI generation: https://ollama.ai/")
        print("3. Run the full pipeline: python run_pipeline.py examples/sample_vault")
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())