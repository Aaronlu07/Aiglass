@echo off
chcp 65001 >nul
title AI Glass System Start
color 0A

echo ========================================================
echo               AI 盲人辅助眼镜系统 一键启动
echo ========================================================
echo.
echo 正在启动核心服务端和 Web 监视器 (端口 8000)...
start "AI Glass Server" cmd /c "python app.py --server"

echo 等待服务端就绪...
timeout /t 3 /nobreak >nul

echo.
echo 正在启动本地客户端 UI 界面...
start "AI Glass UI" cmd /c "python app.py --ui-only"

echo.
echo ========================================================
echo 服务已全部启动！
echo.
echo 1. Web 监视器地址: http://127.0.0.1:8000
echo 2. 本地 UI 界面已在独立窗口打开
echo 3. 请确保 ESP32-CAM 已烧录最新代码并连接到 WiFi
echo ========================================================
echo.
echo 按任意键退出此启动脚本 (不会关闭已启动的服务)...
pause >nul
