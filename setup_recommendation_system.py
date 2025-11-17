"""
Setup script for the recommendation system.
Installs required dependencies and creates necessary directories.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
   [object Object]{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "artifacts/gnn",
        "artifacts/cbf",
        "artifacts/hybrid",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {directory}")
    
    print("✅ All directories created")


def install_dependencies():
    """Install required Python packages."""
    packages = [
        "torch",
        "torch-geometric",
        "sentence-transformers",
        "faiss-cpu",  # Use faiss-gpu if CUDA is available
        "transformers",
        "numpy",
        "scikit-learn",
    ]
    
    print[object Object]nstalling Python packages...")
    print("This may take several minutes...")
    
    for package in packages:
        success = run_command(
            f"{sys.executable} -m pip install {package}",
            f"Installing {package}"
        )
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")
            print("You may need to install it manually")


def check_existing_packages():
    """Check which packages are already installed."""
    print("\n🔍 Checking existing packages...")
    
    packages = {
        "torch": "PyTorch",
        "torch_geometric": "PyTorch Geometric",
        "sentence_transformers": "Sentence Transformers",
        "faiss": "FAISS",
        "transformers": "Transformers",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
    }
    
    installed = []
    missing = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            installed.append(name)
            print(f"  ✓ {name} is installed")
        except ImportError:
            missing.append(name)
            print(f"  ✗ {name} is NOT installed")
    
    return installed, missing


def main():
    """Main setup function."""
    print("="*60)
    pr[object Object]ecommendation System Setup")
    print("="*60)
    
    # Check existing packages
    installed, missing = check_existing_packages()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nDo you want to install missing packages? (y/n): ")
        
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("\n⚠️  Skipping package installation")
            print("Please install the following packages manually:")
            for package in missing:
                print(f"  - {package}")
    else:
        print("\n✅ All required packages are already installed!")
    
    # Create directories
    create_directories()
    
    # Print next steps
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Start your Django server:")
    print("   python manage.py runserver")
    print("\n2. Train the models:")
    print("   # GNN Model")
    print("   curl -X POST http://localhost:8000/api/v1/recommend/gnn/train/ \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"force_retrain\": true, \"sync\": true}'")
    print("\n   # Content-Based Model")
    print("   curl -X POST http://localhost:8000/api/v1/recommend/content/train/ \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"force_retrain\": true, \"sync\": true}'")
    print("\n   # Hybrid Model")
    print("   curl -X POST http://localhost:8000/api/v1/recommend/hybrid/train/ \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"force_retrain\": true, \"sync\": true}'")
    print("\n3. Test recommendations:")
    print("   curl 'http://localhost:8000/api/v1/recommend/gnn/recommend/?user_id=<USER_ID>&product_id=<PRODUCT_ID>'")
    print("\n4. See RECOMMENDATION_SYSTEM_IMPLEMENTATION.md for full documentation")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

