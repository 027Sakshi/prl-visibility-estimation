param(
    [switch]$Apply
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = (Get-Location).Path
$Required = @(
    "run_prl_pipeline.py",
    "src\prl\01_verify_prl_dataset.py",
    "src\prl\04_finetune_fusion.py"
)

foreach ($Path in $Required) {
    if (-not (Test-Path (Join-Path $Root $Path))) {
        throw "Run this script from the PRL project root. Missing: $Path"
    }
}

function Show-Action {
    param([string]$Message)
    $Prefix = if ($Apply) { "APPLY" } else { "PREVIEW" }
    Write-Host "[$Prefix] $Message"
}

function Remove-SafePath {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Show-Action "Remove $Path"
        if ($Apply) {
            Remove-Item -LiteralPath $Path -Recurse -Force
        }
    }
}

function Move-SafePath {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path -LiteralPath $Source) {
        Show-Action "Move $Source -> $Destination"

        if ($Apply) {
            $Parent = Split-Path -Parent $Destination
            New-Item -ItemType Directory -Force -Path $Parent | Out-Null

            if (Test-Path -LiteralPath $Destination) {
                throw "Destination already exists: $Destination"
            }

            Move-Item -LiteralPath $Source -Destination $Destination
        }
    }
}

Write-Host "Project root: $Root"
Write-Host "Mode: $(if ($Apply) { 'apply changes' } else { 'preview only' })"
Write-Host ""

# Clean only source-oriented project folders.
# Deliberately do not traverse venv, .venv, .git, data, models, or results.
$CleanupRoots = @(
    "src",
    "tests",
    "experiments",
    "notebooks",
    "scripts"
)

$CacheDirectoryNames = @(
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints"
)

foreach ($RelativeRoot in $CleanupRoots) {
    $SearchRoot = Join-Path $Root $RelativeRoot

    if (-not (Test-Path -LiteralPath $SearchRoot)) {
        continue
    }

    Get-ChildItem -LiteralPath $SearchRoot -Recurse -Force -Directory `
        -ErrorAction SilentlyContinue |
        Where-Object { $CacheDirectoryNames -contains $_.Name } |
        Sort-Object FullName -Descending |
        ForEach-Object { Remove-SafePath $_.FullName }

    Get-ChildItem -LiteralPath $SearchRoot -Recurse -Force -File `
        -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension -in @(".pyc", ".pyo") -or
            $_.Name -in @(".DS_Store", "Thumbs.db", "desktop.ini")
        } |
        ForEach-Object { Remove-SafePath $_.FullName }
}

# Clean matching machine-generated files only at the repository root.
Get-ChildItem -LiteralPath $Root -Force -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Extension -in @(".pyc", ".pyo") -or
        $_.Name -in @(".DS_Store", "Thumbs.db", "desktop.ini")
    } |
    ForEach-Object { Remove-SafePath $_.FullName }

# Machine-specific inventory contains absolute local paths.
Remove-SafePath (Join-Path $Root "results\prl\result_inventory.csv")

# Move completion notes into the documentation tree.
Move-SafePath `
    (Join-Path $Root "PRL_COMPLETION_RUN_GUIDE.md") `
    (Join-Path $Root "docs\run-guide.md")

Move-SafePath `
    (Join-Path $Root "PRL_COMPLETION_REPORT.md") `
    (Join-Path $Root "docs\project-status.md")

Move-SafePath `
    (Join-Path $Root "docs\PRL_Visibility_Research_Project_Master_Report.docx") `
    (Join-Path $Root "docs\archive\PRL_Visibility_Research_Project_Master_Report.docx")

# Archive superseded experiment entry points instead of deleting provenance.
$LegacyExperiments = @(
    "A1_weather_baseline",
    "A2_dinov2_image",
    "A2_pca",
    "A3_Final",
    "A3_fusion_model",
    "optimization",
    "PCA_Search"
)

foreach ($Name in $LegacyExperiments) {
    Move-SafePath `
        (Join-Path $Root "experiments\$Name") `
        (Join-Path $Root "experiments\legacy\$Name")
}

# Remove only genuinely empty placeholder experiment directories.
foreach ($Name in @("A4_finetuning", "A5_reside_adaptation")) {
    $Path = Join-Path $Root "experiments\$Name"

    if (Test-Path -LiteralPath $Path) {
        $Items = @(Get-ChildItem -LiteralPath $Path -Force)

        if ($Items.Count -eq 0) {
            Remove-SafePath $Path
        } else {
            Write-Host "[KEEP] Non-empty placeholder retained: $Path"
        }
    }
}

Write-Host ""

if (-not $Apply) {
    Write-Host "No files were changed. Review the preview, then run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\prepare_github.ps1 -Apply"
} else {
    Write-Host "Cleanup applied."
    Write-Host "The virtual environment, data, trained models, and generated results were not traversed or deleted."
}