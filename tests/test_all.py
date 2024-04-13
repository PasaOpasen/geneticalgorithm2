

import os
import subprocess



for file in os.listdir('./'):
    if file.endswith('.py') and file != __file__:

        subprocess.call(file, shell=True)

