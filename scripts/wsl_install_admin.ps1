#requires -RunAsAdministrator

[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "Installing WSL2 and distro: $Distro"
try {
    wsl.exe --install -d $Distro
} catch {
    Write-Warning "wsl --install failed, retrying with --web-download"
    wsl.exe --install --web-download -d $Distro
}

Write-Step "Setting WSL default version to 2"
wsl.exe --set-default-version 2

Write-Step "Updating WSL kernel"
wsl.exe --update

Write-Step "Current WSL status"
wsl.exe --status
wsl.exe -l -v

Write-Warning "Restart Windows now."
Write-Warning "After reboot, launch '$Distro' once and create your Linux username/password."
Write-Warning "Then run: scripts\\wsl_bootstrap_from_windows.ps1"
