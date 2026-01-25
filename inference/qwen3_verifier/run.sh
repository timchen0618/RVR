#!/bin/bash

# Submit all round two retrieve jobs
echo "Submitting batch jobs..."

sbatch inference/qwen3_verifier/contriever/round_two_retrieve.sbatch
sbatch inference/qwen3_verifier/contriever/round_two_retrieve_base.sbatch
sbatch inference/qwen3_verifier/infly/round_two_retrieve_base.sbatch
sbatch inference/qwen3_verifier/infly/round_two_retrieve.sbatch
sbatch inference/qwen3_verifier/qwen3/round_two_retrieve.sbatch
sbatch inference/qwen3_verifier/qwen3/round_two_retrieve_base.sbatch

echo "All jobs submitted!"