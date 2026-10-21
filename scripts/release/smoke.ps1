# Native Windows release-artifact smoke test for T4.7; see v1.md §12.
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$DkcPath = ""
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($DkcPath)) {
    $Command = Get-Command dkc -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        throw "dkc is not installed on PATH and -DkcPath was not provided"
    }
    $DkcPath = $Command.Source
}
$DkcPath = (Resolve-Path $DkcPath).Path

$Scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("dkc-release-smoke-" + [guid]::NewGuid())
$HomeDirectory = Join-Path $Scratch "home"
$Fixture = Join-Path $Scratch "repository"
$Binary = Join-Path $Scratch "dkc.exe"
$OldHome = $env:HOME
$OldUserProfile = $env:USERPROFILE
$OldModels = $env:DKC_MODELS_DIR
$OldLocation = Get-Location

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)] [string]$Program,
        [Parameter(Mandatory = $true)] [string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

function Invoke-DkcJson {
    param(
        [Parameter(Mandatory = $true)] [string[]]$Arguments,
        [Parameter(Mandatory = $true)] [string]$Output
    )
    $Lines = & $Binary @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dkc exited with code $LASTEXITCODE"
    }
    $Text = $Lines -join [Environment]::NewLine
    [System.IO.File]::WriteAllText($Output, $Text + [Environment]::NewLine)
    return $Text | ConvertFrom-Json
}

try {
    New-Item -ItemType Directory -Path $Scratch | Out-Null
    New-Item -ItemType Directory -Path $HomeDirectory, $Fixture | Out-Null
    Copy-Item -Path $DkcPath -Destination $Binary
    Copy-Item -Path (Join-Path $RepositoryRoot "fixtures\tiny\*") -Destination $Fixture -Recurse

    $env:HOME = $HomeDirectory
    $env:USERPROFILE = $HomeDirectory
    $env:DKC_MODELS_DIR = Join-Path $Scratch "models"

    Invoke-Native -Program git -Arguments @("-C", $Fixture, "init", "--quiet")
    Invoke-Native -Program git -Arguments @("-C", $Fixture, "add", ".")
    Invoke-Native -Program git -Arguments @(
        "-C", $Fixture,
        "-c", "user.name=Deepkit Release Smoke",
        "-c", "user.email=release-smoke@example.invalid",
        "commit", "--quiet", "-m", "release smoke fixture"
    )

    Set-Location $Fixture
    $null = Invoke-DkcJson -Arguments @("init", "--profile", "local", "--yes", "--json") -Output (Join-Path $Scratch "init.json")
    $Index = Invoke-DkcJson -Arguments @("index", "--level", "1", "--json") -Output (Join-Path $Scratch "index.json")
    $Search = Invoke-DkcJson -Arguments @("search", "should_retry", "--mode", "lexical", "--no-rerank", "--json") -Output (Join-Path $Scratch "search.json")
    $Status = Invoke-DkcJson -Arguments @("status", "--json") -Output (Join-Path $Scratch "status.json")

    if (-not ($Search.hits | Where-Object { $_.path -eq "script.rs" })) {
        throw "release smoke search did not return script.rs"
    }
    if ($Status.files -lt 2 -or $Status.units -lt 1) {
        throw "release smoke status does not show an indexed fixture"
    }
    if ($Index.embedding.state -notin @("quiescent", "skipped")) {
        throw "release smoke embedding pass did not complete"
    }
    if ($Status.queue.dead -ne 0) {
        throw "release smoke status reported dead jobs"
    }

    [ordered]@{
        binary = $Binary
        files = $Status.files
        units = $Status.units
        search_hits = @($Search.hits).Count
        script_hit = $true
    } | ConvertTo-Json -Compress
}
finally {
    Set-Location $OldLocation
    $env:HOME = $OldHome
    $env:USERPROFILE = $OldUserProfile
    $env:DKC_MODELS_DIR = $OldModels
    Remove-Item -Path $Scratch -Recurse -Force -ErrorAction SilentlyContinue
}
