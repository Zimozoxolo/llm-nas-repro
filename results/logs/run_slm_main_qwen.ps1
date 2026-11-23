# Runs slm_main with local Qwen2.5-7B (16K context), seed by seed; report after each seed.
# Safe to re-launch: finished runs are skipped; a failed run is logged and retried on the next launch.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Set-Location "$env:USERPROFILE\llm-nas-repro"
$log = "results\logs\slm_main_qwen.log"
$set = @("--set", "llm.base_url=http://localhost:11434/v1", "llm.model=qwen2.5-7b-16k", "llm.max_tokens=8192", "llm.timeout=3600")
foreach ($seed in 0, 1, 2) {
    python -m llmnas run slm_main --seeds $seed @set 2>&1 | Out-File -Append -Encoding utf8 $log
    python -m llmnas report slm_main @set 2>&1 | Out-File -Append -Encoding utf8 $log
    "=== seed $seed finished $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" | Out-File -Append -Encoding utf8 $log
}
"=== ALL DONE ===" | Out-File -Append -Encoding utf8 $log
