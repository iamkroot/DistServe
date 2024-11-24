# Building DistServe on Delta

# connect to an interactive account 
# srun to bash or zsh

# we will get these from conda
# otherwise, we get into some infinite loop during CMake configure step
module unload nccl cuda cudnn

# clone the project
git clone https://github.com/LLMServe/DistServe.git && cd DistServe

# setup the distserve conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive

# not sure if needed
conda install nccl

# to fix https://github.com/theislab/cellrank/issues/864
conda install openmpi


# MANUAL- edit src/CMakeLists.txt remove unittest, also remove FetchContent of gtest
# this is needed because gtest doesn't build, for whatever reason

# If building on non-A100 nodes, need to specify architecture version
# see https://github.com/NVIDIA/cutlass/issues/4
# -arch=sm_70

# build
cmake -B build && cmake --build build -j$(nproc)

# install distserve via pip
cd ..
pip install -e .


