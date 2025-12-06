## Setup

### Python environment

### Downloading HELM results

```
mkdir -p data/crfm-helm-public/lite/benchmark_output/runs/_all

magnet download helm --benchmark=lite --list-versions | while read version; do
    magnet download helm data/crfm-helm-public --benchmark=lite --version="$version" --runs "regex:med_qa.*"
    (cd data/crfm-helm-public/lite/benchmark_output/runs/_all && ln -s "../$version"/* .)
done
```
