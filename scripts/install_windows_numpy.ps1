param(
    [string]$KritaInstallDir = "",
    [string]$KritaResourceDir = "",
    [string]$Python = "py",
    [string]$PythonVersion = "",
    [string]$NumpyRequirement = "numpy>=1.26,<3",
    [switch]$Upgrade
)

$ErrorActionPreference = "Stop"

function Find-KritaPythonVersion {
    $installDirs = @()
    if ($KritaInstallDir) {
        $installDirs += $KritaInstallDir
    }
    $installDirs += @(
        "C:\Program Files\Krita (x64)",
        "C:\Program Files\Krita"
    )

    foreach ($installDir in $installDirs) {
        $binDir = Join-Path $installDir "bin"
        if (-not (Test-Path $binDir)) {
            continue
        }

        foreach ($pythonDll in Get-ChildItem -Path $binDir -Filter "python*.dll" -File) {
            if ($pythonDll.Name -match "^python(?<major>\d)(?<minor>\d{2})\.dll$") {
                $major = $Matches["major"]
                $minor = [int]$Matches["minor"]
                return "$major.$minor"
            }
        }
    }

    return ""
}

function Invoke-TargetPython {
    param([string[]]$Arguments)

    $pythonArgs = @()
    if ($Python -eq "py" -and $PythonVersion) {
        $pythonArgs += "-$PythonVersion"
    }

    & $Python @pythonArgs @Arguments
}

if (-not $KritaResourceDir) {
    if (-not $env:APPDATA) {
        throw "APPDATA is not set. Pass -KritaResourceDir explicitly."
    }
    $KritaResourceDir = Join-Path $env:APPDATA "krita"
}

if (-not $PythonVersion) {
    $PythonVersion = Find-KritaPythonVersion
}

if (-not $PythonVersion) {
    throw "Could not detect Krita's embedded Python version. Pass -PythonVersion, for example -PythonVersion 3.10."
}

$vendorRoot = Join-Path $KritaResourceDir "oklab_colour_picker"
$vendorDir = Join-Path $vendorRoot "site-packages"
New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

Write-Host "Using Krita resource directory: $KritaResourceDir"
Write-Host "Using Python launcher target: $PythonVersion"
Write-Host "Installing $NumpyRequirement into: $vendorDir"

Invoke-TargetPython -Arguments @(
    "-c",
    "import platform, sys; print(sys.version); assert platform.architecture()[0] == '64bit', 'Krita for Windows requires a 64-bit Python for NumPy wheels'"
)

$pipArgs = @(
    "-m",
    "pip",
    "install",
    "--only-binary=:all:",
    "--target",
    $vendorDir,
    $NumpyRequirement
)

if ($Upgrade) {
    $pipArgs += "--upgrade"
}

Invoke-TargetPython -Arguments $pipArgs

Invoke-TargetPython -Arguments @(
    "-c",
    "import sys; sys.path.insert(0, r'$vendorDir'); import numpy; print('Installed NumPy', numpy.__version__)"
)

Write-Host ""
Write-Host "Done. Restart Krita so the OKLab Colour Selector can load NumPy."
