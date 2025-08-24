#!/usr/bin/env python3
"""
Test runner for S3 Vectors tests.
"""

import sys
import os
import subprocess
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_unit_tests():
    """Run all unit tests."""
    print("🧪 Running unit tests...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/unit", 
            "-v", 
            "--tb=short"
        ], cwd=os.path.join(os.path.dirname(__file__), ".."), check=True)
        print("✅ Unit tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Unit tests failed!")
        return False


def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Running integration tests...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/integration", 
            "-v", 
            "--tb=short"
        ], cwd=os.path.join(os.path.dirname(__file__), ".."), check=True)
        print("✅ Integration tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Integration tests failed!")
        return False


def run_all_tests():
    """Run all tests."""
    print("🧪 Running all tests...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests", 
            "-v", 
            "--tb=short"
        ], cwd=os.path.join(os.path.dirname(__file__), ".."), check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Some tests failed!")
        return False


def run_coverage():
    """Run tests with coverage."""
    print("📊 Running tests with coverage...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests", 
            "--cov=src/app", 
            "--cov-report=html:htmlcov", 
            "--cov-report=term-missing",
            "-v"
        ], cwd=os.path.join(os.path.dirname(__file__), ".."), check=True)
        print("✅ Coverage tests completed!")
        print("📁 Coverage report saved to htmlcov/")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Coverage tests failed!")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="S3 Vectors Test Runner")
    parser.add_argument(
        "test_type", 
        choices=["unit", "integration", "all", "coverage"],
        default="all",
        nargs="?",
        help="Type of tests to run"
    )
    
    args = parser.parse_args()
    
    print("🔍 S3 Vectors Test Runner")
    print("=" * 40)
    
    success = False
    
    if args.test_type == "unit":
        success = run_unit_tests()
    elif args.test_type == "integration":
        success = run_integration_tests()
    elif args.test_type == "coverage":
        success = run_coverage()
    else:  # all
        success = run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()