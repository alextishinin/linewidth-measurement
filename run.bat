@echo off
rem Launch the linewidth app (Mode button switches Live / Single sweep).
"%LocalAppData%\Programs\Python\Python312\python.exe" "%~dp0linewidth_live.py" %*
pause
