








INSTALLATION:
Notable python packages required: pycuda, moderngl, pyglet

Installing pycuda:
Pycuda should be able to compile CUDA code (ie: you need nvcc). Also you need a device which is CUDA capable such as a Nvidia GPU.
Thankfuly, it is easy to install (at least on Windows), but you need to follow the instructions to make sure everything is setup correctly.
Before installing pycuda, install the CUDA development toolkit. Check the installation of the cuda compiler with "nvcc --version"  before installing pycuda.
Windows being Windows, you might need to download visual Studio (not VS code) in order to have the binaries that will be used by nvcc, so, if on Windows, the order of install should be (Visual studio -> install the C/C++ things from there (easy to find in their interface)) -> then install the CUDA dev toolkit (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ and https://developer.nvidia.com/cuda-downloads)  -> then  check that nvcc is installed, then, install pyCUDA (for isntance with pip).
The CUDA compiler is quite easy to install on some Linux distributions too.


PAPER:
Paper: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-203.pdf
"Estimated neighbour sets and smoothed sampled global interactions are sufficient for a fast approximate t-SNE." by Pierre Lambert, Edouard Couplet, Cyril de Bodt, and John A. Lee
