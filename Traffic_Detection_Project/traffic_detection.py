import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Print current working directory to verify where we are
    print(f"Current working directory: {os.getcwd()}")

    # List contents of current directory
    print("\nContents of current directory:")
    print(os.listdir())

    # Let's try to find our data directories
    current_dir = Path.cwd()
    print("\nSearching for dataset directories...")
    
    # Search for train, valid, and test directories
    for item in current_dir.rglob("*"):
        if item.is_dir() and item.name in ['train', 'valid', 'test']:
            print(f"Found {item.name} directory at: {item}")
            # List first few files in directory
            files = list(item.glob('*'))[:5]
            if files:
                print(f"Sample files in {item.name}:")
                for f in files:
                    print(f"  {f.name}")
            else:
                print(f"No files found in {item.name}")
            print()

if __name__ == "__main__":
    main()