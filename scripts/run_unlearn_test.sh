#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run unlearning progression test
echo "Testing unlearning progression..."
python3 scripts/testing/test_unlearning_progression.py \
    --log_dir "checkpoints/unlearn_cat_chks_xbh7dm4p?" \
    --output_dir "unlearning_test_results" \
    --device "cuda" \
    --seed 42

echo "Unlearning test completed!"
echo "Check the 'unlearning_test_results' directory for results."
