Benchmark data lives here but is not committed. Re-create it with

    python -m llmnas prepare-data

which downloads (~2 GB, mostly the NATS-Bench archive) and writes
`processed/nb201_{cifar10,cifar100,ImageNet16-120}.csv` (~5 MB each).
Set `LLMNAS_DATA=/some/path` to keep the data elsewhere (e.g. Google Drive on Colab).
