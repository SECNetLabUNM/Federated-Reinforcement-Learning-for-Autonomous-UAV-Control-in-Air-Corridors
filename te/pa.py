import sys
import os

def check_sys_path(path_to_check):
    # Check if path is current directory
    if path_to_check == '':
        print("Path is the current directory.")
        return

    # Check PYTHONPATH
    pythonpath = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if path_to_check in pythonpath:
        print(f"Path is in PYTHONPATH environment variable.")
        return

    # Check for .pth files
    for p in sys.path:
        if os.path.isdir(p):
            for file in os.listdir(p):
                if file.endswith('.pth'):
                    with open(os.path.join(p, file), 'r') as f:
                        paths = f.read().splitlines()
                        if path_to_check in paths:
                            print(f"Path is included in {file} file.")
                            return

    print("Path origin not found. It might be a standard library, site-packages directory, or added programmatically.")

path_to_check = '/home/kun/PycharmProjects/air-corridor/air_corridor'
check_sys_path(path_to_check)