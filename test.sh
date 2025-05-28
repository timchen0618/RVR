srun --pty -c 2 --mem=40GB --gres=gpu:a100:1 /bin/bash # request an RTX8000 GPU node

SINGULARITY_IMAGE=/scratch/work/public/singularity/ubuntu-20.04.1.sif

OVERLAY_FILE=/scratch/work/public/examples/greene-getting-started/overlay-15GB-500K-pytorch.ext3
## Note: this overlay file has PyTorch installed using command
## conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

## Run interactively on the GPU node with singularity exec --nv
singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE /bin/bash
source /ext3/miniconda3/bin/activate

# ## Exit the Singularity session
# Singularity> exit

# ## Exit interactive session
# exit