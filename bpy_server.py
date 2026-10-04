import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
dependency_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".blender_deps")
if os.path.isdir(dependency_dir):
    sys.path.insert(0, dependency_dir)

from src.server.bpy_server import run

def main():
    run()

if __name__ == "__main__":
    main()
