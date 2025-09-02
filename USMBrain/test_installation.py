"""
Test script to verify USM Brain installation.
Run this script to check if all dependencies are properly installed.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'numpy',
        'faiss',
        'openai',
        'pandas',
        'sentence_transformers',
        'tqdm',
        'torch',
        'transformers',
        'scikit-learn',
        'scipy'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} packages failed to import:")
        for package in failed_imports:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    print("=" * 50)
    
    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✅ NumPy: Created array {arr}")
        
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformers: Model loaded successfully")
        
        # Test FAISS
        import faiss
        dimension = 384  # all-MiniLM-L6-v2 output dimension
        index = faiss.IndexFlatL2(dimension)
        print("✅ FAISS: Index created successfully")
        
        # Test OpenAI (without API key)
        import openai
        print("✅ OpenAI: Package imported successfully")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("USM Brain Installation Test")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Run tests
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! USM Brain is ready to use.")
            print("\nNext steps:")
            print("1. Set your OpenAI API key in config.py or .env file")
            print("2. Run: python USMBrain.py")
        else:
            print("\n⚠️  Some functionality tests failed. Check the errors above.")
    else:
        print("\n❌ Installation incomplete. Please fix the import errors above.")

if __name__ == "__main__":
    main()
