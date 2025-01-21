I. MOTIVATION:

I.a. Neighbour embeddings

Neighbour embeddings, such as t-SNE and UMAP, are powerfull tools to reduce the dimensionality of data nonlinearly, these algorithms are particularly good at mitigating the effects of concentration of norms, which tends to happen in high dimensions. The low-dimensional (LD) representation fo the high-dimensional (HD) data can be used for data visualisation if the target LD dimensionality is 2 or 3, or as a prepocessing step for further computations: a preprocessing with tSNE/UMAP can help denoise data and reduce the effects of the "curses of dimensionality" on downstream algorithms, for isntance for clsutering purposes.

To keep things very short and intuitive, neighbour embeddings typically work in 2 phases: first, KNN sets in HD are determined for each points. Then, a LD representation of the points are moved around in the embedding in the aim to preserve the KNN sets computed in HD: if the neighbour sets in LD are similar to those in HD, one can say that the LD representation captures well the local structures in the data. The points in LD are subject to attractive and repulsive forces, intuitively, they are attracted to their neighbours in HD, and repulsed from the points that are close in LD but not in their neighbour sets in HD.Kernel functions in both spaces are used as surrogates for KNN sets, mainly for derivability. The kernel function in LD can have heavier tails that the one in HD to encourage the separation of clusters in the embedding, but this can also lead to artificial tearing of the manifold.

I.b. The t-SNE family vs the UMAP family

UMAP and t-SNE are some of the most commonly used neighbour embedding algorithms, they both have adavantages and shortcommings. 

tSNE (and its ancestor SNE introduced by the nobel price G. Hinton in 2002), has the advantage of modelling quite precisely the local repulsive interactions between points, allowing greater precision in small scales (small neigbhourhoods), to achieive these precise local repulsive interactions, tSNE models the LD space, slowing the algorithm considerably. Accelerated t-SNE methods have been introduced, such as Barnes-Hut tSNE using partitionning trees and FFT-accelerated Interpolation-based t-SNE (FIt-SNE), while these accelerations render t-SNE applicable to large datasets (hundreds of thousands or millions of points), their effective speed is still inferior to alternatives that do not model the LD space (such as UMAP), and the modelling of the LD space restrict their use to very low dimensions for the embedding space, limiting their use to data visualisation. Some versions of t-SNE allow for more flexibility in the 

UMAP is a more recent algorithm that is also widely used in the data visualisation community. Contrary to t-SNE, it doesn't explicitely consider local repulsive forces between points in the embedding and only models global repulsions using random sampling of points across the dataset. This allows for very fast iterations and also removes the constraint on the dimensionality of the embedding space, opening the field of neighbour embeddings to other tasks than data visualisation. Claims have been done that UMAP better preserves the global structure of the data, but these are yet to be verified.

I.c What this method brings

The purpose of this method is to bring the best of both worlds: we propose a t-SNE algorithm with flexible LD kernel "tail heaviness" and explicit local repulsive interactions, while achieving a compuational complexity of O(N) (like UMAP and FIt-SNE), but with an effective speed that is much faster than the current t-SNE algorithms. This method is implemented on GPU, so it would not be fair to compare its speed to other multithreaded CPU implementations, but the simplicity of our algorithm has a low number of computations at each iteration, regardless of the harware.
Our method doesn't constrain the embedding space to low dimensionalities, and removes the 2-phase approach of neighbour embeddings (1: compute HD KNN, 2: optimise the LD embedding) this renders the method more interative: the user can change a hyperparameter in HD (perplexity, distance metric) and the method responds immediately to the change. In theory, the method can easily adapt to new points, but this functionality is not implemented at the moment.
We recomend using this method with the built-in GUI.

II. Why interactivity is key in neighbour embeddings

Let's take the example of the handwritten MNIST dataset. This dataset is 28x28 pixel images of handwritten digits from 0 to 9, written in white on a black background. One can ask: does this dataset have structures? If so, do some substructures separate from the rest and form clusters? Do structures appear at different scales? Since we all know what digits are, we might expect to see 10 clusters in the data, at least on a certain scale. tSNE and UMAP with their default hyperparameters do indeed tend to show 10 clusters, coloured here by their label to facilitate visualisation.


However, some people write "1" as a single straight line, some write this straight angled to the right, others add a small oblique line on the top.

<img width="901" alt="image" src="https://github.com/user-attachments/assets/a3ee8346-dc3f-4e2e-bebc-43d03d103dc7" />



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
