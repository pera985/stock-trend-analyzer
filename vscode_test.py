#!/usr/bin/env python3
"""
Simple test to run in VSCode
This will show which Python VSCode is using and if numpy works
"""

import sys
import os

print("=" * 70)
print("VSCODE PYTHON TEST")
print("=" * 70)
print()
print("Which Python am I using?")
print(f"  Executable: {sys.executable}")
print(f"  Version: {sys.version.split()[0]}")
print()
print("Where am I running from?")
print(f"  Current directory: {os.getcwd()}")
print()
print("Testing numpy import...")

try:
    import numpy as np
    print(f"✅ SUCCESS - numpy {np.__version__}")
    print(f"   Installed at: {np.__file__}")
except Exception as e:
    print(f"❌ FAILED - {e}")
    print()
    print("This means VSCode is using a different Python or environment")
    print("that doesn't have numpy installed.")
    print()
    print("To fix in VSCode:")
    print("1. Press Cmd+Shift+P")
    print("2. Type 'Python: Select Interpreter'")
    print("3. Choose: /Library/Developer/CommandLineTools/usr/bin/python3")

print()
print("=" * 70)
