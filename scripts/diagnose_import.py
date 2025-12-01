"""
Diagnose import issues with UFLDv2
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UFLDV2_PATH = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"

sys.path.insert(0, str(UFLDV2_PATH))

print("DIAGNOSING UFLDv2 IMPORTS")
print(f"UFLDv2 Path: {UFLDV2_PATH}")
print()

# Check what files exist
print("Files in model directory:")
model_dir = UFLDV2_PATH / "model"
if model_dir.exists():
    for f in sorted(model_dir.iterdir()):
        print(f"  ✓ {f.name}")
else:
    print("  ✗ Model directory not found!")
print()

# Try importing step by step
print("Testing imports:")
print("-" * 60)

# Test 1: Import torch
try:
    import torch
    print("✓ torch imported successfully")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ Failed to import torch: {e}")

# Test 2: Try to read model_culane.py to see what it imports
print("\nChecking model_culane.py imports:")
model_culane_path = UFLDV2_PATH / "model" / "model_culane.py"
if model_culane_path.exists():
    print(f"✓ Found model_culane.py")
    with open(model_culane_path, 'r') as f:
        lines = f.readlines()
        print("\n  First 30 lines of model_culane.py:")
        for i, line in enumerate(lines[:30], 1):
            print(f"  {i:3d}: {line.rstrip()}")
else:
    print(f"✗ model_culane.py not found at {model_culane_path}")

# Test 3: Try importing the model
print("\n" + "-" * 60)
print("Attempting to import parsingNet:")
try:
    from model.model_culane import parsingNet
    print("✓ Successfully imported parsingNet!")
    print(f"  Type: {type(parsingNet)}")
except ImportError as e:
    print(f"✗ Failed to import parsingNet")
    print(f"  Error: {e}")
    print(f"  Error type: {type(e).__name__}")
    
    # Try to get more details
    import traceback
    print("\n  Full traceback:")
    traceback.print_exc()
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)