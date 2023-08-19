#/bin/bash
# Fetch mobileone CoreML models from model zoo host.
location=https://docs-assets.developer.apple.com/ml-research/datasets/mobileone
for model_number in 0 1 2 3 4; do
  model=mobileone_s$model_number.mlmodel
  curl $location/$model -o $model;
done

