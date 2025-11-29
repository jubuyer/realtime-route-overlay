"""
Verify that all required components are properly set up
"""
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_directory(path, name):
    """Check if directory exists"""
    if path.exists():
        print(f"✓ {name}: {path}")
        return True
    else:
        print(f"✗ {name}: {path} (NOT FOUND)")
        return False

def check_file(path, name):
    """Check if file exists"""
    if path.exists():
        print(f"✓ {name}: {path}")
        return True
    else:
        print(f"✗ {name}: {path} (NOT FOUND)")
        return False

def check_ufldv2_structure():
    """Check UFLDv2 repository structure"""
    ufldv2_path = PROJECT_ROOT / "models" / "Ultra-Fast-Lane-Detection-v2"
    
    print("\n" + "="*60)
    print("Checking UFLDv2 Repository Structure")
    print("="*60)
    
    required_files = [
        "model/model.py",
        "model/__init__.py",
        "utils/common.py",
        "configs/tusimple_res18.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = ufldv2_path / file_path
        exists = check_file(full_path, f"  {file_path}")
        all_exist = all_exist and exists
    
    return all_exist

def main():
    print("="*60)
    print("SETUP VERIFICATION")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check directories
    print("\n" + "="*60)
    print("Checking Directories")
    print("="*60)
    
    all_good = True
    
    # Models directory
    models_dir = PROJECT_ROOT / "models"
    all_good &= check_directory(models_dir, "Models directory")
    
    # UFLDv2 repository
    ufldv2_dir = models_dir / "Ultra-Fast-Lane-Detection-v2"
    all_good &= check_directory(ufldv2_dir, "UFLDv2 repository")
    
    # Weights directory
    weights_dir = models_dir / "Ultra-Fast-Lane-Detection-v2" / "weights"
    all_good &= check_directory(weights_dir, "Weights directory")
    
    # Data directory
    data_dir = PROJECT_ROOT / "datasets" / "TUSimple"
    all_good &= check_directory(data_dir, "TuSimple dataset")
    
    # Results directory
    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print(f"  Creating results directory...")
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Results directory: {results_dir}")
    else:
        check_directory(results_dir, "Results directory")
    
    # Check model weights
    print("\n" + "="*60)
    print("Checking Model Weights")
    print("="*60)
    
    res18_path = weights_dir / "tusimple_res18.pth"
    res34_path = weights_dir / "tusimple_res34.pth"
    
    has_res18 = check_file(res18_path, "ResNet-18 weights")
    has_res34 = check_file(res34_path, "ResNet-34 weights")
    
    if not (has_res18 or has_res34):
        print("\n No model weights found!")
        print("Download from: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2")
    
    # Check UFLDv2 structure
    if ufldv2_dir.exists():
        ufldv2_ok = check_ufldv2_structure()
        all_good &= ufldv2_ok
    else:
        print("\n UFLDv2 not cloned!")
        print("Run: git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git models/Ultra-Fast-Lane-Detection-v2")
        all_good = False
    
    # Check dataset
    print("\n" + "="*60)
    print("Checking Dataset")
    print("="*60)
    
    if data_dir.exists():
        clips_dir = data_dir / "test_set"/"clips"
        if clips_dir.exists():
            images = list(clips_dir.rglob("*.jpg"))
            print(f"✓ Found {len(images)} images in dataset")
        else:
            print(f"✗ Clips directory not found: {clips_dir}")
            all_good = False
    else:
        print(f"✗ TuSimple dataset not found")
        print("Download from: https://www.kaggle.com/datasets/manideep1108/tusimple")
        all_good = False
    
    # Final status
    print("\n" + "="*60)
    if all_good and (has_res18 or has_res34):
        print("✓ ALL CHECKS PASSED - Ready to run!")
        print("="*60)

    else:
        print("✗ SETUP INCOMPLETE")
        print("="*60)
        print("\nPlease fix the issues above before running inference.")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)