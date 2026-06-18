param(
    [Parameter(Mandatory = $true)]
    [string]$ApiKey
)

if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Error "ApiKey cannot be empty."
    exit 1
}

[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", $ApiKey, "User")
$env:DEEPSEEK_API_KEY = $ApiKey

Write-Host "DEEPSEEK_API_KEY has been saved to the Windows User environment."
Write-Host "Restart Antigravity IDE / terminal so OpenCode can read the new environment variable."
Write-Host "Then verify with: opencode debug config"
