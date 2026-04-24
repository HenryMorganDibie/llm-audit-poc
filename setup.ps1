# setup.ps1
# Run this in PowerShell to scaffold the project locally
# Usage: .\setup.ps1

$folders = @(
    "audit",
    "data",
    "scripts",
    "results",
    "docs"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "Created: $folder" -ForegroundColor Green
    } else {
        Write-Host "Exists:  $folder" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Project structure ready." -ForegroundColor Cyan
Write-Host "Next: pip install -r requirements.txt" -ForegroundColor Cyan
