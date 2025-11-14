# Script để xóa cache Python (.pyc và __pycache__)
# Chỉ xóa trong source code, không xóa trong venv

Write-Host "Đang tìm và xóa cache Python..." -ForegroundColor Yellow

# Tìm và xóa các thư mục __pycache__ (trừ venv)
$pycacheDirs = Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -notlike "*\venv\*" -and $_.FullName -notlike "*\.git\*" }

if ($pycacheDirs.Count -gt 0) {
    Write-Host "Tìm thấy $($pycacheDirs.Count) thư mục __pycache__" -ForegroundColor Cyan
    foreach ($dir in $pycacheDirs) {
        Write-Host "  Xóa: $($dir.FullName)" -ForegroundColor Gray
        Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Đã xóa tất cả thư mục __pycache__" -ForegroundColor Green
} else {
    Write-Host "Không tìm thấy thư mục __pycache__ trong source code" -ForegroundColor Green
}

# Tìm và xóa các file .pyc (trừ venv)
$pycFiles = Get-ChildItem -Path . -Include *.pyc -Recurse -File -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -notlike "*\venv\*" -and $_.FullName -notlike "*\.git\*" }

if ($pycFiles.Count -gt 0) {
    Write-Host "Tìm thấy $($pycFiles.Count) file .pyc" -ForegroundColor Cyan
    foreach ($file in $pycFiles) {
        Write-Host "  Xóa: $($file.FullName)" -ForegroundColor Gray
        Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Đã xóa tất cả file .pyc" -ForegroundColor Green
} else {
    Write-Host "Không tìm thấy file .pyc trong source code" -ForegroundColor Green
}

Write-Host "`nHoàn thành! Bây giờ hãy restart Django server." -ForegroundColor Yellow

