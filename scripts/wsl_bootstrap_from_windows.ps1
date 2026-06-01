[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$RepoPathWindows = "C:\Users\suchy\Studia\Masters\RR\LSTM_SGARCH_rr"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $RepoPathWindows)) {
    throw "Repository path does not exist: $RepoPathWindows"
}

$drive = $RepoPathWindows.Substring(0, 1).ToLowerInvariant()
$rest = $RepoPathWindows.Substring(2).Replace("\", "/")
$repoPathWsl = "/mnt/$drive$rest"

Write-Host "Running WSL setup for repo: $repoPathWsl" -ForegroundColor Cyan
wsl.exe -d $Distro -- bash -lc "cd '$repoPathWsl' && bash scripts/wsl_setup.sh"
