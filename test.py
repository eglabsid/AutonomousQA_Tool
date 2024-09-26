import os
import platform
from functools import wraps

def os_specific_task(os_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_os = platform.system()
            if current_os == os_name:
                return func(*args, **kwargs)
            else:
                print(f"Skipping {func.__name__} on {current_os}")
        return wrapper
    return decorator

@os_specific_task("Windows")
def windows_task():
    print("Running on Windows")
    os.system('dir')

@os_specific_task("Linux")
def linux_task():
    print("Running on Linux")
    os.system('ls')

@os_specific_task("Darwin")
def macos_task():
    print("Running on macOS")
    os.system('ls')

def perform_os_specific_tasks():
    windows_task()
    linux_task()
    macos_task()

perform_os_specific_tasks()
