# Duplicate-fix test, paid GPT-4.1 arms (approved 2026-11-23, cap $23; PREREGISTRATION_duplicate_fix.md Section 8).
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
foreach ($arm in "arm_a", "arm_b") {
    $p = Start-Process python -ArgumentList "-m llmnas run dupfix_arms --variants $arm" -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "results\logs\dupfix_$arm.out.log" -RedirectStandardError "results\logs\dupfix_$arm.err.log"
    "$arm finished at $(Get-Date -Format s), exit code $($p.ExitCode)" | Out-File -Append -Encoding utf8 results\logs\dupfix_arms.done.log
}
