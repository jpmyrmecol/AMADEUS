@REM Copyright (C) 2026 Yusuke Notomi
@REM SPDX-License-Identifier: AGPL-3.0-only

@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv\Scripts\python.exe was not found.
    echo [INFO] This file must be placed in AMADEUS\tools\check_torch.cmd
    exit /b 1
)

echo ===== AMADEUS environment diagnostic =====
".venv\Scripts\python.exe" -c "import numpy, scipy, torch, torchvision; print('numpy =', numpy.__version__); print('scipy =', scipy.__version__); print('torch =', torch.__version__); print('torchvision =', torchvision.__version__); print('torch CUDA runtime =', torch.version.cuda or 'CPU'); print('CUDA available =', torch.cuda.is_available()); print('GPU =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('torch path =', torch.__file__); print('torchvision path =', torchvision.__file__)"
if errorlevel 1 exit /b 1

echo.
echo ===== SciPy binary compatibility test =====
".venv\Scripts\python.exe" -c "import numpy as np; from scipy.ndimage import gaussian_filter1d; x=np.array([0.,1.,0.]); y=gaussian_filter1d(x,1.0); assert np.isfinite(y).all(); print('SciPy ndimage = OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] NumPy/SciPy compatibility test failed.
    exit /b 1
)

echo.
echo ===== torchvision CUDA NMS test =====
".venv\Scripts\python.exe" -c "import torch; from torchvision.ops import nms; assert torch.cuda.is_available(), 'CUDA is not available'; b=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda'); s=torch.tensor([0.9,0.8],device='cuda'); print('NMS result =', nms(b,s,0.5).cpu().tolist()); torch.cuda.synchronize(); print('torchvision CUDA NMS = OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch/torchvision CUDA test failed.
    exit /b 1
)

echo.
echo [OK] AMADEUS numerical and CUDA dependencies are working.
endlocal
