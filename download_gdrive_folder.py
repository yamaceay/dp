#!/usr/bin/env python3
"""
Download a folder from Google Drive using gdown.

Usage:
    python download_gdrive_folder.py <folder_url_or_id> [output_dir]
    
Examples:
    # Using folder URL
    python download_gdrive_folder.py "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j"
    
    # Using folder ID
    python download_gdrive_folder.py "1a2b3c4d5e6f7g8h9i0j" ./my_downloads
    
    # With custom output directory
    python download_gdrive_folder.py "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j" ./data
"""

import sys
import os
import gdown


def extract_folder_id(url_or_id: str) -> str:
    """Extract folder ID from URL or return ID if already in ID format."""
    if "drive.google.com" in url_or_id:
        # Extract ID from URL
        # Format: https://drive.google.com/drive/folders/FOLDER_ID
        if "/folders/" in url_or_id:
            folder_id = url_or_id.split("/folders/")[1].split("?")[0].split("/")[0]
            return folder_id
        else:
            raise ValueError(f"Invalid Google Drive folder URL: {url_or_id}")
    else:
        # Assume it's already a folder ID
        return url_or_id


def download_folder(folder_url_or_id: str, output_dir: str = None, quiet: bool = False) -> str:
    """
    Download a folder from Google Drive.
    
    Args:
        folder_url_or_id: Google Drive folder URL or ID
        output_dir: Output directory (default: current directory)
        quiet: Suppress progress output
        
    Returns:
        Path to downloaded folder
    """
    try:
        folder_id = extract_folder_id(folder_url_or_id)
        
        if output_dir is None:
            output_dir = os.getcwd()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Downloading folder ID: {folder_id}")
        print(f"Output directory: {output_dir}")
        
        # Download the folder
        # gdown.download_folder expects a URL or ID
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        result = gdown.download_folder(
            url=url,
            output=output_dir,
            quiet=quiet,
            use_cookies=False,
            remaining_ok=True
        )
        
        print(f"\n✓ Download complete!")
        print(f"Files saved to: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"Error downloading folder: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing folder URL or ID", file=sys.stderr)
        sys.exit(1)
    
    folder_url_or_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    download_folder(folder_url_or_id, output_dir)


if __name__ == "__main__":
    main()
