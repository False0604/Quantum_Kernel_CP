"""
Bridge runner to allow executing streamlit run app.py directly from the project root.
"""
import os
import sys
from pathlib import Path

pkg_dir = Path(__file__).resolve().parent / "VS Code" / "ML-Learning"
sys.path.insert(0, str(pkg_dir))
os.chdir(pkg_dir)

# Execute the primary Streamlit app
with open(pkg_dir / "app.py", encoding="utf-8") as f:
    code = compile(f.read(), str(pkg_dir / "app.py"), "exec")
    exec(code, globals())
