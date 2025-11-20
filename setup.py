"""
Setup and validation script for Legal Knowledge Graph project
"""
import sys
from pathlib import Path

print("=" * 80)
print("Legal Knowledge Graph - Setup & Validation")
print("=" * 80)

# Step 1: Check Python version
print("\n[1/6] Checking Python version...")
if sys.version_info < (3, 8):
    print("✗ Python 3.8+ required")
    sys.exit(1)
print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Step 2: Check if .env file exists
print("\n[2/6] Checking environment configuration...")
env_file = Path(__file__).parent / ".env"
if not env_file.exists():
    print("⚠ .env file not found")
    print("\nCreating .env file from template...")
    env_example = Path(__file__).parent / ".env.example"
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✓ .env file created")
        print("\n⚠ IMPORTANT: Edit .env file and add your API keys!")
        print("   Required:")
        print("   - GEMINI_API_KEY")
        print("   - NEO4J_PASSWORD")
    else:
        print("✗ .env.example not found")
else:
    print("✓ .env file exists")

# Step 3: Validate configuration
print("\n[3/6] Validating configuration...")
try:
    from config.settings import validate_config
    validate_config()
    print("✓ Configuration valid")
except ImportError as e:
    print(f"⚠ Cannot import config module: {e}")
    print("  This is expected if dependencies aren't installed yet")
except Exception as e:
    print(f"⚠ Configuration validation: {e}")
    print("  Please check your .env file")

# Step 4: Check data files
print("\n[4/6] Checking data files...")
data_dir = Path(__file__).parent

files_to_check = {
    "FIR_DATASET.csv": "FIR Dataset (IPC Sections)",
    "legal_acts_chunks.csv": "Legal Acts",
    "the_constitution_of_india.pdf": "Constitution PDF",
    "Object_casedocs": "Case Documents Directory"
}

for file_name, description in files_to_check.items():
    file_path = data_dir / file_name
    if file_path.exists():
        if file_path.is_dir():
            count = len(list(file_path.glob("*.txt")))
            print(f"✓ {description}: {count} files")
        else:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✓ {description}: {size_mb:.1f} MB")
    else:
        print(f"✗ {description}: Not found at {file_path}")

# Step 5: Test Neo4j connection (if dependencies installed)
print("\n[5/6] Testing Neo4j connection...")
try:
    from config.neo4j_config import test_connection
    if test_connection():
        print("✓ Neo4j connection successful")
    else:
        print("✗ Neo4j connection failed")
        print("\n  Troubleshooting:")
        print("  1. Make sure Neo4j Desktop is running")
        print("  2. Check NEO4J_PASSWORD in .env file")
        print("  3. Verify NEO4J_URI in .env file (default: bolt://localhost:7687)")
except ImportError:
    print("⚠ Cannot test Neo4j (dependencies not installed)")
except Exception as e:
    print(f"✗ Neo4j connection error: {e}")

# Step 6: Test Gemini API (if dependencies installed)
print("\n[6/6] Testing Gemini API...")
try:
    import google.generativeai as genai
    from config.settings import GEMINI_API_KEY
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("✗ GEMINI_API_KEY not set in .env file")
        print("\n  Get your API key from: https://makersuite.google.com/app/apikey")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello")
        print("✓ Gemini API connection successful")
except ImportError:
    print("⚠ Cannot test Gemini API (dependencies not installed)")
except Exception as e:
    print(f"✗ Gemini API error: {e}")
    print("  Please check your GEMINI_API_KEY in .env file")

# Final summary
print("\n" + "=" * 80)
print("Setup Validation Complete")
print("=" * 80)

print("\n📋 Next Steps:")
print("  1. Install dependencies: pip install -r requirements.txt")
print("  2. Configure .env file with your API keys")
print("  3. Ensure Neo4j Desktop is running")
print("  4. Run: python main.py --max-cases 300")
print("\n💡 Quick Start:")
print("  python setup.py          # Run this script")
print("  pip install -r requirements.txt")
print("  python main.py --help    # See all options")
print("  python main.py --max-cases 300 --skip-constitution")

print("\n" + "=" * 80)
