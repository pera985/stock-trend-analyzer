#!/usr/bin/env python3
"""
Find out which Python VSCode is using
"""
import sys
import subprocess

print("=" * 70)
print("WHICH PYTHON IS VSCODE USING?")
print("=" * 70)
print()
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# Try to install numpy to THIS Python
print("Attempting to install numpy to THIS Python interpreter...")
print("=" * 70)
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--user', 'numpy'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print()
print("Now testing if numpy imports...")
try:
    import numpy
    print(f"✅ SUCCESS! numpy {numpy.__version__}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print()
    print("The Python VSCode is using:", sys.executable)
    print("Copy this path and we'll install packages to it.")
