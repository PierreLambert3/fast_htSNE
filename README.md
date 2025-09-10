

<img src="https://github.com/user-attachments/assets/e4286f33-0dfe-415a-9212-6443ea794b81" alt="croped2" width="400">

Preprint available at:
https://arxiv.org/abs/2509.07681#:~:text=9%20Sep%202025%5D-,FUnc%2DSNE%3A%20A%20flexible%2C%20Fast%2C%20and,Unconstrained%20algorithm%20for%20neighbour%20embeddings&text=Neighbour%20embeddings%20(NE)%20allow%20the,to%20handle%20very%20large%20datasets.

I. MOTIVATION:

I.a. Neighbour embeddings

Neighbour embeddings, such as t-SNE and UMAP, are powerful tools to reduce the dimensionality of data nonlinearly. These algorithms are particularly good at mitigating the effects of the concentration of norms, which tends to happen in high dimensions. The low-dimensional (LD) representation of the high-dimensional (HD) data can be used for data visualization if the target LD dimensionality is 2 or 3 or as a preprocessing step for further computations. Preprocessing with t-SNE/UMAP can help denoise data and reduce the effects of the "curse of dimensionality" on downstream algorithms, for instance, for clustering purposes.

To keep things very short and intuitive, neighbour embeddings typically work in two phases: first, KNN sets in HD are determined for each point. Then, an LD representation of the points is moved around in the embedding to preserve the KNN sets computed in HD. If the neighbour sets in LD are similar to those in HD, one can say that the LD representation captures the local structures in the data well. The points in LD are subject to attractive and repulsive forces. Intuitively, they are attracted to their neighbours in HD and repulsed from the points that are close in LD but not in their neighbour sets in HD. Kernel functions in both spaces are used as surrogates for KNN sets, mainly for differentiability. The kernel function in LD can have heavier tails than the one in HD to encourage the separation of clusters in the embedding, but this can also lead to artificial tearing of the manifold.

I.b. The most common neighboru embedding algorithms: t-SNE and UMAP

UMAP and t-SNE are some of the most commonly used neighbour embedding algorithms; they both have advantages and shortcomings.

t-SNE (and its ancestor SNE, introduced by the Nobel Prize winner G. Hinton in 2002) has the advantage of modeling local repulsive interactions between points quite precisely, allowing greater local precision. To achieve these precise local repulsive interactions, t-SNE computes the full KNN set in the LD space for its original verison, or models the LD space itself in most accelerated versions. The optimisation process is iterative, so computing the KNN set/modeling the LD space needs to be done at each iteration, slowing the algorithm considerably. The most common accelerated t-SNE methods are Barnes-Hut t-SNE, which uses partitioning trees, and FFT-accelerated Interpolation-based t-SNE (FIt-SNE). While these accelerations make t-SNE applicable to large datasets (hundreds of thousands or millions of points), their effective speed is still inferior to alternatives that do not model the LD space (such as UMAP). Additionally, the modeling of the LD space restricts its use to very low-dimensional embedding spaces, limiting its application to data visualization. Some versions of t-SNE allow for more flexibility in the choice of the kernel used in LD to define the neighbourhoods, shifting the dynamics of the forces at hand, which in term produces embeddings that preserve different structures in the data (an example is shown later).

UMAP is a more recent algorithm that is also widely used in the data visualization community. Unlike t-SNE, it does not accurately consider local repulsive forces between points in the embedding and only models global repulsions using random sampling of points across the dataset. This allows for very fast iterations and also removes the constraint on the dimensionality of the embedding space, opening the field of neighbour embeddings to applications beyond data visualization, there is however a heavy a cost in the quality of the preserved by the algorithm, as observed in the paper accompanying this method ("FUnc-SNE: A flexible, Fast, and Unconstrained algorithm for neighbour embeddings"), in [3], and in [4]. Claims have been made that UMAP better preserves the global structure of the data, but these are yet to be verified and where sometimes the result of different initialisations of the embeddings (random vs PCA/Laplacian eigenmap).

I.c What this method brings

The purpose of this work is to bring the best of both worlds: we propose a t-SNE algorithm with flexible LD kernel "tail heaviness" and explicit local repulsive interactions while achieving a computational complexity of O(N) (in line with the fastest algorithms) but with an effective speed that is higher than current t-SNE algorithms. This method is implemented on GPU, so it would not be fair to compare its speed to other multithreaded CPU implementations, but the simplicity of our algorithm results in a low number of computations at each iteration, comparable to UMAP, regardless of the hardware.

Our method does not constrain the embedding space to low dimensionalities and removes the two-phase approach of neighbour embeddings (1: compute HD KNN, 2: optimize the LD embedding). This makes the method more interactive: the user can change a hyperparameter in HD (perplexity, distance metric), and the method responds immediately to the change. In theory, the method can easily adapt to new points, but this functionality is not implemented at the moment.

We recommend using this method with the built-in GUI.

II. Why interactivity is key in neighbour embeddings

Let's take the example of the handwritten MNIST dataset. This dataset consists of 28x28 pixel monochrome images of handwritten digits from 0 to 9, on a black background. One can ask: does this dataset have structure? If so, do some substructures separate from the rest and form clusters? Do structures appear at different scales?

Since we all know what digits are, we might expect to see 10 clusters in the data, at least at a certain scale. t-SNE and UMAP, with their default hyperparameters, do indeed tend to show 10 clusters. Here is a t-SNE embedding using default hyperparameters, where the observations are colored by their label to facilitate visualization.

<img width="307" alt="tsneparams_mnist" src="https://github.com/user-attachments/assets/0c206494-5e54-4e78-bd0b-f087d47440e1" />

However, some people write "1" as a single straight line, some write it angled to the right, and others add a small oblique line on the top. We cannot see these distinctions in the embedding presented above because the MNIST dataset, like most datasets, has an intrinsic dimensionality larger than 2 (around 6 for MNIST, if memory serves right). This means that whatever dimensionality reduction method we use, if we embed the dataset into only 2 or 3 dimensions, the original structures will not be perfectly preserved in the embedding: the algorithm will render only part of the information contained in the dataset, and discard the rest. What is kept in the embeddings and what is discarded depends on the dataset, and on the method that models it (along with its hyperparameters). 

In this case, t-SNE, with its chosen hyperparameter set, has "decided" to preserve the part of the structure that clusters the same digits together. This configuration is the most stable one that the optimization process found, and adding finer- or coarser-grained structures to the embedding would have caused an unacceptable increase in the loss function being optimized. Changing the algorithm and some hyperparameters can drive the embedding toward other configurations, highlighting different structures in the data, as shown in the following figure. A mode detailed explanation and analysis is performed in the paper. In short, the algorithm has separated clusters by tearing the manifold along zones of weakness, visualised here as a dip in the probability ditribution if projecting the HD data of two clusters along the direction that separate their centre of gravity. 

<img width="552" height="466" alt="Screenshot 2025-09-10 105701" src="https://github.com/user-attachments/assets/b31f3a01-534a-449e-9d33-0c4be89047b0" />

When using dimensionality reduction (DR) techniques on real datasets, we generally want to explore the data, meaning we do not know precisely in advance what to look for. Using DR is therefore a delicate art where the user must not interpret what is shown as the unique true structure of the data but rather as one partly correct, yet potentially misleading, representation. This requires a careful balance between confirmation bias (filtering out irrelevant information) and the ability to question one's assumptions about the data to explore the dataset in a balanced manner.

The proposed method allows for seamless and fast transitions between many different embedding configurations, enabling a more rigorous visualization of the data.

Below are some MNIST digit embeddings using classical neighbour embedding techniques. The colours are not standardized across embeddings.

<img width="1562" alt="together_others" src="https://github.com/user-attachments/assets/f112d5bc-34f7-4ce2-a6e0-e9a5af607d4e" />

Now, here is a subset of possible configurations using the proposed method, passing from one to another is a matter of seconds.

<img width="1608" alt="together_others - Copy" src="https://github.com/user-attachments/assets/2cb37ece-c3dd-4eb3-ba46-ed4a0da58732" />

On another dataset from the domain of a single-cell transcriptomics: the gene expressions are evaluated by looking at messenger RNA in cells found in rat brains. More details in [2]. We see that the global structures are more apparent using light LD kernel tails, and that smaller scale structures can be explored across multiple granularities using heavier tails.

<img width="1251" alt="rna_high_attrac" src="https://github.com/user-attachments/assets/28731fb4-fdd0-4302-b5ba-7777e24ce301" />

III. The method in action

All these gifs show the method running on a laptop with a modern GPU, therefore the GPU is throttled. To fit the size constraint of this markdown, every 3rd frame in the gifs is dropped, making them faster than the true speed, however the actual effective speed on the laptop is close to what is shown here. Additional compression was done, lowering the quality of the animated images.

Left: MNIST train set (60k points, 50 dimensions: we took the 50 principal components), from the first iteration of the algorithm. This shows that the KNN search is fast and quickly leads to a usable visualisation:
Right : Single cell RNAseq (a bit more than 20k points, also the first 50 PC).

<img src="https://github.com/user-attachments/assets/65d083d0-dcef-4bf6-97ba-874c8e441513" alt="croped2" width="500">
<img src="https://github.com/user-attachments/assets/e4286f33-0dfe-415a-9212-6443ea794b81" alt="croped2" width="400">

Left: Coil-20. These are rotating images: in the HD space each object should lay in a ring manifold corresponding to one full rotation.
Right: Abalone, this dataset without preprocessing is particularily suited to show differences between Euclidean distances in HD and cosine distances

<img src="https://github.com/user-attachments/assets/d108d6e5-caaa-4eb2-bd8f-be1b6cfb7303" alt="croped2" width="400">
<img src="https://github.com/user-attachments/assets/8a51d9dd-066b-42c7-bab2-3cc3a609f1bb" alt="croped2" width="400">

A quick demonstration of the algorithm at work with and embedding dimensionality of 6. This is not useful for visualisation, but the embedding can be used for a downstream machine learning task. The embeddings can be produced with the gui (on closing the windows, the embedding is returned, or you can click "save" to save to disk), or they can be produced without the gui, however, as hopefully demonstrated in the gifs above, we recomend exploring the data using the GUI.

![2025-01-24-15-26-30-ezgif com-crop](https://github.com/user-attachments/assets/ba70b0e3-3070-4f8d-92cd-fe05b7c34a03)

The method can also be applied to very large datasets, as here on embeddings of half of imagenet (a bit more than 600k observations). This method works with larger datasets (more than 1M points), but then the slowdown becomes important and the interactivity is diminished. On reasonble machines, a couple of hundred of thousands of points stays enjoyable in terms of interactivity. In the two following images, the data is taken from the latent space of large vision-language transformer models, the top row shows representations of the latent space of the EVA model, a very large neural network. The second row shows representations from a model called ViT B 16, a smaller model. In supervised tasks, the large model performs better, and we can see that the tSNE embeddings tend to show better class separation as well. 

Some zones in the embeddings have better class separation than others, in both models. It could be interesting to explore the data further, to find why some zones get a better 2D representation than others: is it because of a locally larger intrinsic dimensionality in the latent space? Or perhaps the concepts are harder and the neural network didn't find an organised representation there? Perhaps these correspond to concepts less frequent in pre training? The fast tSNE embeddings can be useful to explore these hypotheses, in conjunction with other tools. One might for instance colour the data points depending on their class error in testing, or depending on frequencies of concepts in pre training to help develop intuition. One could also filter out some points to only focus on part of the embedding.

![big](https://github.com/user-attachments/assets/aaed0459-cd65-4629-be92-9e78e2d8fc2d)

![Capture d’écran 2025-01-29 101849 - Copy](https://github.com/user-attachments/assets/9f7b4f30-1606-494e-affe-132b89b52e83)

INSTALLATION:

Notable python packages required: pycuda, moderngl, pyglet

General guidelines:

Pycuda should be able to compile CUDA code (ie: you need nvcc). Also you need a device which is CUDA capable such as a Nvidia GPU.
Thankfuly, it is easy to install (at least on Windows), but you need to follow the instructions to make sure everything is setup correctly.
Before installing pycuda, install the CUDA development toolkit. Check the installation of the cuda compiler with "nvcc --version"  before installing pycuda.
Windows being Windows, you might need to download Visual Studio (not VS code) in order to have the binaries that will be used by nvcc, so, if on Windows, the order of install should be (Visual studio -> install the C/C++ things from there (easy to find in their interface)) -> then install the CUDA dev toolkit (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ and https://developer.nvidia.com/cuda-downloads)  -> then  check that nvcc is installed, then, install pyCUDA (for isntance with pip).
The CUDA compiler is quite easy to install on some Linux distributions too.



Linux: 


Here is a text dump that a colleague using Linux used to properly install everything, it's likely generated text:
"
To get PyCUDA and ModernGL working with CUDA 12.8 on Ubuntu 24.04, follow this complete setup process:

1. Install NVIDIA Driver (with EGL and OpenGL support)
Install driver version 535, which supports CUDA 12.8 and provides the necessary libraries:
 
    sudo apt install nvidia-driver-535 libnvidia-gl-535 libglvnd-dev
 
Note: The package libnvidia-egl-535 may not be available — that’s fine. The two above are sufficient.
 
After installation, reboot:
 
    sudo reboot
 
2. Install CUDA 12.8 (Toolkit only)
Download the `.run` (local) installer from the official NVIDIA CUDA Downloads page:
 
    https://developer.nvidia.com/cuda-downloads
 
Choose:
- OS: Linux
- Architecture: x86_64
- Distribution: Ubuntu 22.04 (works fine on 24.04)
- Installer Type: runfile (local)
 
Then run the installer:
 
    chmod +x cuda_12.8.*.run
    sudo ./cuda_12.8.*.run --toolkit --override
 
During the interactive setup:
- Say NO to installing the driver
- Say YES to installing the toolkit
 
3. Set Environment Variables
Add CUDA 12.8 to your PATH by editing your ~/.bashrc:
 
    export PATH="/usr/local/cuda-12.8/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
 
Apply the changes:
 
    source ~/.bashrc
 
4. Install Python Libraries
Using a Conda environment is recommended (e.g., Miniconda). Activate your environment:
 
    conda activate myEnvironement
 
Install the required packages:
 
    pip install pycuda moderngl pyglet
 
5. Verify Installation
- Check CUDA version:
 
    nvcc --version
 
- Check NVIDIA driver:
 
    nvidia-smi
 
- Test PyCUDA:
 
    python -c "import pycuda.driver as cuda; import pycuda.autoinit; print('CUDA Runtime Version:', cuda.get_version())"



References:

[1] Kobak, D., Linderman, G., Steinerberger, S., Kluger, Y., Berens, P. (2020). Heavy-Tailed Kernels Reveal a Finer Cluster Structure in t-SNE Visualisations. In: Brefeld, U., Fromont, E., Hotho, A., Knobbe, A., Maathuis, M., Robardet, C. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2019. Lecture Notes in Computer Science(), vol 11906. Springer, Cham. https://doi.org/10.1007/978-3-030-46150-8_8

[2] Kobak D, Berens P. The art of using t-SNE for single-cell transcriptomics. Nat Commun. 2019 Nov 28;10(1):5416. doi: 10.1038/s41467-019-13056-x

[3] "Low-dimensional embeddings of high-dimensional data" at https://arxiv.org/abs/2508.15929

[4] "SQuadMDS: A lean Stochastic Quartet MDS improving global structure preservation in neighbor embedding like t-SNE and UMAP" at https://www.sciencedirect.com/science/article/abs/pii/S0925231222008402


PAPER:

Intruductory paper: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-203.pdf
"Estimated neighbour sets and smoothed sampled global interactions are sufficient for a fast approximate t-SNE." by Pierre Lambert, Edouard Couplet, Cyril de Bodt, and John A. Lee

More complete paper: https://arxiv.org/abs/2509.07681#:~:text=9%20Sep%202025%5D-,FUnc%2DSNE%3A%20A%20flexible%2C%20Fast%2C%20and,Unconstrained%20algorithm%20for%20neighbour%20embeddings&text=Neighbour%20embeddings%20(NE)%20allow%20the,to%20handle%20very%20large%20datasets.
"FUnc-SNE: A flexible, Fast, and Unconstrained algorithm for neighbour embeddings" by Pierre Lambert, Edouard Couplet, Michel Verleysen, John Aldo Lee


