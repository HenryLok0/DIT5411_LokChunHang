@echo off
echo Checking WSL training status...
wsl -d Ubuntu-22.04 -u root -- bash /mnt/c/Users/ihenr/Downloads/DIT5411/scripts/status.sh
pause
