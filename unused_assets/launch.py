import subprocess
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
script = os.path.join(current_dir, 'quick_autolabel.py')

print(f"Launching script: {script}")
print(f"Script exists: {os.path.exists(script)}")

proc = subprocess.Popen([sys.executable, script], 
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,
                       universal_newlines=True,
                       cwd=current_dir)

for line in proc.stdout:
    print(line, end='')

proc.wait()
sys.exit(proc.returncode)
