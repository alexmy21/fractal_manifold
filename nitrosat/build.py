#!/usr/bin/env python3
"""
Build script for NitroSAT shared library.

Usage:
    python build.py           # Build the library
    python build.py clean     # Clean build artifacts
    python build.py install   # Build and copy to system path
    
Based on NitroSAT by Sethu Iyer (sethuiyer95@gmail.com)
Wrapper for fractal_manifold project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Get the directory containing this script
NITROSAT_DIR = Path(__file__).parent.absolute()
LIB_NAME = "libnitrosat.so"
LIB_NAME_MAC = "libnitrosat.dylib"


def get_platform():
    """Detect platform."""
    if sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "win32":
        return "windows"
    return "unknown"


def build():
    """Build the NitroSAT shared library."""
    platform = get_platform()
    
    source_file = NITROSAT_DIR / "nitrosat_api.c"
    
    if platform == "macos":
        output_file = NITROSAT_DIR / LIB_NAME_MAC
        cmd = [
            "gcc", "-O3", "-fPIC", "-shared",
            "-march=native",
            "-std=c99",
            "-o", str(output_file),
            str(source_file),
            "-lm"
        ]
    elif platform == "linux":
        output_file = NITROSAT_DIR / LIB_NAME
        cmd = [
            "gcc", "-O3", "-fPIC", "-shared",
            "-march=native",
            "-std=c99",
            "-o", str(output_file),
            str(source_file),
            "-lm"
        ]
    else:
        print(f"Unsupported platform: {platform}")
        return False
    
    print(f"Building NitroSAT for {platform}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=NITROSAT_DIR)
        if result.returncode != 0:
            print(f"Build failed:\n{result.stderr}")
            return False
        print(f"✓ Built successfully: {output_file}")
        return True
    except FileNotFoundError:
        print("Error: gcc not found. Please install GCC compiler.")
        return False


def clean():
    """Clean build artifacts."""
    artifacts = [
        NITROSAT_DIR / LIB_NAME,
        NITROSAT_DIR / LIB_NAME_MAC,
        NITROSAT_DIR / "nitrosat",  # standalone executable
    ]
    for artifact in artifacts:
        if artifact.exists():
            artifact.unlink()
            print(f"Removed: {artifact}")
    
    # Clean __pycache__
    pycache = NITROSAT_DIR / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
        print(f"Removed: {pycache}")
    
    print("✓ Cleaned")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "clean":
            clean()
        elif cmd == "install":
            if build():
                print("Library built. No system-wide install needed - use from this directory.")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python build.py [clean|install]")
    else:
        build()


if __name__ == "__main__":
    main()
