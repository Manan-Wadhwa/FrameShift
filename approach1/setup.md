₹# FrameShift - Simple Setup

This is a minimal setup for running the `v1.py` demo on Windows (PowerShell).

1. Create a virtual environment

```powershell
python -m venv .\venv
```

2. Activate the environment (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

3. Install required packages

```powershell
pip install opencv-python-headless numpy matplotlib scikit-image
```

4. Run the demo

```powershell
python v1.py
```

Notes:
- If you prefer using conda, create and activate a conda env instead.
- In VS Code you can run cells marked with `#%%` using the Python extension.

That's it — simple and quick.
