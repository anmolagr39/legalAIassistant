#!/usr/bin/env python3
"""
Setup and Installation Script for Legal RAG System
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_files():
    """Check if data files exist."""
    base_dir = os.path.dirname(__file__)
    constitution_path = os.path.join(base_dir, "the_constitution_of_india (1).pdf")
    legal_acts_path = os.path.join(base_dir, "legal-acts.csv")
    
    missing_files = []
    
    if not os.path.exists(constitution_path):
        missing_files.append("the_constitution_of_india (1).pdf")
    
    if not os.path.exists(legal_acts_path):
        missing_files.append("legal-acts.csv")
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure these files are in the legal-rag-system directory.")
        return False
    
    print("✅ All data files found!")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/processed",
        "data/chroma_db"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Directories created!")

def main():
    print("🏛️ Legal RAG System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the legal-rag-system directory")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data files
    if not check_data_files():
        return
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add your Gemini API key to the .env file")
    print("2. Run: python main.py --setup")
    print("3. Run: python main.py --interactive")

if __name__ == "__main__":
    main()