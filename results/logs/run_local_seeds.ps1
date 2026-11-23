# Local-LLM seeds 1-2 on CIFAR-10, six devices (no API cost: Ollama on localhost only).
# Resumable: finished runs are skipped, so re-running this script continues where it stopped.
# Restarted 2026-11-20 (the first pass stopped after 9 of 24 runs; its output was not captured).
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
"Start at $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\logs\local_seeds.log
foreach ($m in "deepseek-coder-v2-16b-8k", "qwen2.5-7b-8k") {
    $out = "results\logs\local_seeds_$m.out.log"; $err = "results\logs\local_seeds_$m.err.log"
    $p = Start-Process python -ArgumentList "-m llmnas run llm_devices --seeds 1 2 --set llm.model=$m llm.base_url=http://localhost:11434/v1 llm.timeout=3600" -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    "$m finished at $(Get-Date -Format s), exit code $($p.ExitCode)" | Out-File -Append -Encoding utf8 results\logs\local_seeds.log
}
"Done at $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\logs\local_seeds.log
