try:
    import flask
    print("Flask: OK")
except Exception as e:
    print(f"Flask: {e}")

try:
    import ultralytics
    print("Ultralytics: OK")
except Exception as e:
    print(f"Ultralytics: {e}")

try:
    from flask_cors import CORS
    print("CORS: OK")
except Exception as e:
    print(f"CORS: {e}")

try:
    import torch
    print("Torch: OK")
except Exception as e:
    print(f"Torch: {e}")
