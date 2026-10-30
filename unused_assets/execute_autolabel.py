import sys
import os
import importlib.util

script_path = r"c:\Users\PRO\Desktop\root canal\Root-canal-AI-diagnostics\run_autolabel_internal.py"

spec = importlib.util.spec_from_file_location("run_autolabel_internal", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
