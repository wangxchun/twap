@echo off
setlocal enabledelayedexpansion

REM 定義要執行的 num_days 參數
for %%d in (2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) do (
    echo Running with --num_days %%d
    python process_random_days.py --num-days %%d
    python process_market_price.py --num-days %%d
    python process_avg_price.py --num-days %%d
)

echo All tasks completed.
pause

