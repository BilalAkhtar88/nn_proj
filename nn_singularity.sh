singularity shell --bind /gpfs/helios/projects/neuralnet_course/track02/dataset:$(pwd)/dataset:ro \
                  /gpfs/helios/projects/neuralnet_course/track02/containers/pytorch_26.04-py3.sif
