import os
import sys
import subprocess
import shutil
import requests
import argparse
import hashlib
import hmac
import tempfile
from pathlib import Path

# Dependencies required for the current environment (ComfyUI/Venv)
CORE_DEPS = [
    "requests", "python-box", "einops", "omegaconf", "lightning", "addict",
    "fast-simplification", "trimesh", "open3d", "gradio", "bottle", "tornado"
]

# Dependencies required specifically for the Blender standalone server
BLENDER_DEPS = ["bottle", "requests", "scipy", "trimesh", "tornado"]

def find_blender_python(blender_path):
    """Finds the internal python executable relative to the blender executable."""
    blender_bin = Path(blender_path)
    root_dir = blender_bin.parent
    if os.name == 'nt':
        patterns = ["**/python/bin/python.exe", "python/bin/python.exe", "**/python.exe"]
    else:
        patterns = ["**/bin/python3*", "bin/python3*", "**/python3*"]
    
    for pattern in patterns:
        matches = list(root_dir.glob(pattern))
        # Filter out configuration scripts and directories
        matches = [m for m in matches if not str(m).endswith("-config") and m.is_file()]
        if not matches: continue
        
        # Prioritize exact names like 'python3' or 'python.exe'
        exact_names = ["python", "python3", "python.exe"]
        for name in exact_names:
            for m in matches:
                if m.name == name: return str(m)
        
        return str(matches[-1])
    return None

def install_core_section():
    print("\n--- Phase 0: Core Library Setup ---")
    
    # Try to install bpy optionally
    print("Checking for 'bpy' (optional native support)...")
    try:
        import bpy
        print("SUCCESS: Native 'bpy' is already installed.")
    except ImportError:
        print("Attempting to install 'bpy' (optional, might fail)...")
        # Capture output to avoid "red error" text in the console if it fails
        res = subprocess.run([sys.executable, "-m", "pip", "install", "bpy"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0:
            print("SUCCESS: Native 'bpy' installed.")
        else:
            print("NOTICE: 'bpy' (native module) not available. Using system Blender instead.")
            print("        (This is normal and expected on most systems).")

    print(f"Installing required core dependencies: {', '.join(CORE_DEPS)}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install"] + CORE_DEPS, check=True)
        print("SUCCESS: Core dependencies installed.")
    except Exception as e:
        print(f"ERROR: Failed to install core dependencies: {e}")

def install_blender_section():
    print("\n--- Phase 1: Blender Standalone Server Setup ---")
    final_blender = shutil.which("blender")
    
    # Common Windows installation paths as fallbacks
    if not final_blender and os.name == 'nt':
        common_paths = [
            "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.3\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.1\\blender.exe",
            "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                final_blender = p
                break
    
    if not final_blender:
        print("WARNING: 'blender' executable not found in system PATH.")
        print("         The standalone 3D server setup will be skipped.")
        return

    print(f"Blender found at: {final_blender}")
    py_exe = find_blender_python(final_blender)
    if not py_exe:
        print(f"ERROR: Could not find internal Python in Blender path: {final_blender}")
        return

    print(f"Found Blender Python: {py_exe}")
    print(f"Installing server dependencies: {', '.join(BLENDER_DEPS)}...")
    try:
        # Blender's bundled Python has no pip on Windows until ensurepip runs
        subprocess.run([py_exe, "-m", "ensurepip"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([py_exe, "-m", "pip", "install"] + BLENDER_DEPS, check=True)
        print("SUCCESS: Blender server dependencies installed.")
    except Exception as e:
        print(f"ERROR: Failed to install Blender dependencies: {e}")

def install_flash_attn_section(wheel=None, expected_sha256=None):
    print("\n--- Phase 2: Optional Flash Attention Setup ---")
    if wheel is None and expected_sha256 is None:
        print("Keeping existing Flash Attention. SDPA is used when it is unavailable.")
        print("To install a verified wheel, supply --flash-attn-wheel and --flash-attn-sha256.")
        return
    if not wheel or not expected_sha256:
        raise ValueError("Both the wheel and its independently verified SHA-256 are required")
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Invalid SHA-256")
    with tempfile.TemporaryDirectory(prefix="skintokens_wheel_") as directory:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(wheel)
        if parsed.scheme and parsed.scheme not in ("https",):
            # A Windows drive letter is a local path, not a URL scheme.
            if not Path(wheel).is_file():
                raise ValueError("Wheel URLs must use HTTPS")
        filename = unquote(parsed.path.rsplit("/", 1)[-1]) if parsed.scheme == "https" else Path(wheel).name
        if not filename.startswith("flash_attn-") or not filename.endswith(".whl") or Path(filename).name != filename:
            raise ValueError("Expected a Flash Attention wheel filename")
        target = Path(directory) / filename
        if parsed.scheme == "https":
            with requests.get(wheel, stream=True, timeout=60) as response:
                response.raise_for_status()
                if urlparse(response.url).scheme != "https":
                    raise ValueError("Wheel download redirected away from HTTPS")
                with target.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
        else:
            shutil.copyfile(wheel, target)
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ValueError("Flash Attention wheel SHA-256 mismatch; installation refused")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(target)], check=True)
        print("SUCCESS: Verified Flash Attention wheel installed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-attn-wheel", help="Local wheel or HTTPS URL")
    parser.add_argument("--flash-attn-sha256", help="Independently verified wheel SHA-256")
    args = parser.parse_args()
    if bool(args.flash_attn_wheel) != bool(args.flash_attn_sha256):
        parser.error("Supply both --flash-attn-wheel and --flash-attn-sha256")
    print("=== SkinTokens: Complete Installation Script ===")
    install_core_section()
    install_blender_section()
    install_flash_attn_section(args.flash_attn_wheel, args.flash_attn_sha256)
    print("\nInstallation process finished.")

if __name__ == "__main__":
    main()
