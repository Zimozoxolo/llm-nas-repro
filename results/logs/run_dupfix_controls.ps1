# Duplicate-fix controls (free, CPU only): 4 controls x 6 devices x 10 seeds; finished runs are skipped.
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
$p = Start-Process python -ArgumentList "-m llmnas run dupfix_controls" -NoNewWindow -Wait -PassThru -RedirectStandardOutput results\logs\dupfix_controls.out.log -RedirectStandardError results\logs\dupfix_controls.err.log
"dupfix_controls finished at $(Get-Date -Format s), exit code $($p.ExitCode)" | Out-File -Append -Encoding utf8 results\logs\dupfix_controls.done.log
