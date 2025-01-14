import numpy as np

DATA_ROOT = './datasets/'

def fetch_MNIST(atrificially_inflate_n_times=1):
    # binary dump from a C flattened array of floats (32bits) these are the first 50 PCs of the MNIST dataset
    mnist_binaries_path = DATA_ROOT + r"\MNIST_50PC"
    mnist_X_filename = mnist_binaries_path + r"\MNIST_PCA_X.bin"
    mnist_Y_filename = mnist_binaries_path + r"\MNIST_PCA_Y.bin"
    N = 60*1000
    M = 50

    # load the dataset
    X = np.fromfile(mnist_X_filename, dtype=np.float32).reshape(N, M)
    Y = np.fromfile(mnist_Y_filename, dtype=np.float32).reshape(N, 1)

    # allocate new dataset
    N_new = N * atrificially_inflate_n_times
    X_new = np.zeros((N_new, M), dtype=np.float32)
    Y_new = np.zeros((N_new, 1), dtype=np.float32)
    perms = np.random.permutation(N)
    for i in range(atrificially_inflate_n_times):
        X_new[i*N:(i+1)*N, :] = X[perms, :]
        Y_new[i*N:(i+1)*N, :] = Y[perms, :]

    # shuffle the dataset
    perms = np.random.permutation(N_new)
    X_new = X_new[perms, :]
    Y_new = Y_new[perms, :]

    # add some noise
    X_new += np.random.normal(0, 5.0, X_new.shape)

    return N_new, M, X_new, Y_new.astype(np.uint32)

def get_blobs(n=50000, m=32):
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples= n, n_features=m, centers=5, cluster_std=6.0, center_box=(-10.0, 10.0), shuffle=True)
    N, M = X.shape
    return N, M, X.astype(np.float32), Y.astype(np.uint32)

def get_coil20():
    from scipy.io import loadmat
    mat = loadmat(DATA_ROOT + "COIL20.mat")
    X, Y = mat['X'], mat['Y']
    Y = (Y.astype(int) - 1).reshape((-1,))
    N, M = X.shape
    return N, M, X.astype(np.float32), Y.astype(np.uint32)

def get_abalone():
    csvfile = DATA_ROOT + "abalone.csv" # 1,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15   first is the label
    with open(csvfile, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        M = len(lines[0].split(',')) - 1
        X = np.zeros((N, M), dtype=np.float32)
        Y = np.zeros((N, 1), dtype=np.int32)
        for i, line in enumerate(lines):
            parts = line.split(',')
            Y[i] = int(parts[0])
            X[i, :] = np.array(parts[1:], dtype=np.float32)
    return N, M, X.astype(np.float32), (Y+1).astype(np.uint32)

def get_airfoil_self_noise():
    csvfile = DATA_ROOT + "airfoil_noise.csv" # 800;0;0.3048;71.3;0.00266337;126.201   last is the label (regression)
    with open(csvfile, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        M = len(lines[0].split(';')) - 1
        X = np.zeros((N, M), dtype=np.float32)
        Y = np.zeros((N, 1), dtype=np.float32)
        for i, line in enumerate(lines):
            parts = line.split(';')
            Y[i] = float(parts[-1])
            X[i, :] = np.array(parts[:-1], dtype=np.float32)
    return N, M, X.astype(np.float32), Y.astype(np.float32) # float32 for regression, uint32_t for classification!

def get_satellite():
    csvfile = DATA_ROOT + "satellite.csv" # 72;89;94;76;72;89;98;76;76;94;98;76;76;87;91;67;71;87;87;70;71;83;87;67;75;87;93;67;71;87;89;67;71;79;81;62;4
    with open(csvfile, 'r') as f:
        lines = f.readlines()
        N = len(lines)
        M = len(lines[0].split(';')) - 1
        X = np.zeros((N, M), dtype=np.float32)
        Y = np.zeros((N, 1), dtype=np.uint32)
        for i, line in enumerate(lines):
            parts = line.split(';')
            Y[i] = int(parts[-1])
            X[i, :] = np.array(parts[:-1], dtype=np.float32)
    return N, M, X.astype(np.float32), Y.astype(np.uint32)

def get_RNAseq20k():
    xpath = DATA_ROOT + "RNAseq_N20K.npy"
    XY = np.load(xpath)
    RNAcolors = np.load('datasets/RNAseq_colors.npy')
    rgb_colors = []
    for c in RNAcolors:
        rgb_colors.append(np.array(list(int(str(c).lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))) # taken from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb_colors = np.array(rgb_colors)

    X = XY[:, :-1]
    Y =  XY[:, -1]

    perms = np.arange(X.shape[0])
    np.random.shuffle(perms)
    X = X[perms]
    Y = Y[perms]
    N = X.shape[0]
    M = X.shape[1]

    # Y becomes shape (N, 3) with the RBG value corresponding to the class
    # 1. dictionary to map the class to the RGB value
    class_to_rgb = {}
    n_labels = len(np.unique(Y))
    for i in range(n_labels):
        class_to_rgb[i] = rgb_colors[i]
    Y_new  = np.zeros((N, 3), dtype=np.int32)
    for i in range(N):
        Y_new[i] = class_to_rgb[Y[i]]
    Y = Y_new

    return N, M, X.astype(np.float32), Y.astype(np.float32) / 256.0

def run_demo():
    
    from fast_htSNE.fast_htSNE import htSNE

    # 0. A tiny dataset to get a feeling of the algorithm: the training set of MNIST or COIL20. Notice the impact of the "kernel alpha" hyperparameter, be aware that a high "kernel alpha "means lower tails in the LD kernel (closer to SNE)
    N, M, X, Y = get_coil20()
    Xld, _ = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y)

    # a small transcriptomics dataset with custom colours for the classes
    N, M, X, Y = get_RNAseq20k()
    Xld, _ = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y)


    # 1. watch how for some data distribultions/preprocessing types, changing the dist metric can have a large impact (try cosine).
    N, M, X, Y = get_satellite()
    N, M, X, Y = get_abalone()
    Xld, _ = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y)
    
    # 2. example usage for a reasonably large dataset (100k points or less): default hyperparameters
    # You will notice a first warmup period where hyperparameters are automatically changed across time. After that, feel free to change the hyperparameters in the GUI to see how the algorithm responds to different values.
    # Notice how increasintg the "kernel alpha" (ie: decreasing the tails of the kernel in LR: getting closer to SNE than t-SNE) helps the points move around, same for the attraction multiplier.
    # it can often help to vary these  hyperparameters to get a feeling of how the points "want" to move around in the embedding space, and allowing them to move around, before settling to a final hyperparameter setting.
    N, M, X, Y = fetch_MNIST(atrificially_inflate_n_times=1)  # 60k points, 50 principal components
    # N, M, X, Y = fetch_MNIST(atrificially_inflate_n_times=5)   # 5 * 60k points (noisy clones of the original dataset)
    Xld, _ = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y)

    # 3. embedding to more than 2 dimensions: useless for visualisation, but can be a powerfull preprocessing step for a downstream task (supervised task with a loww number of variables, clustering, compression, ...)
    # notice that only the first 6 components will be displayed, but you can chek by saving/loading the embedding that there are indeed 8 components that were computed.
    N, M, X, Y = fetch_MNIST(atrificially_inflate_n_times=1)  # 60k points, 50 principal components
    Xld, _ = htSNE(n_components=8, verbose=True, with_gui=True).fit(X, Y) 
    

    # 4. example usage for a very large dataset: increase the base attraction/repulsion ratio, and also set a large attraction in the GUI and a high kernel alpha (ie: low tails: closer to SNE that the Student-t verions) also high learning rate
    # The higher the number of points, the harder the dimensionality reduction task (after all, the  intrinsic dimensionality of the data can only grow with the number of observations). We suggest doing filtering to avoid having more than, say, 300k points.
    # There are general rules of thumbs to help hyperparameter tuning, but their efective impact dependa a lot on the actual distribution of the data. Therefore, we highly recommend experimenting using the GUI, to find get an idea of the different strucutres of the data, en how the algorithm respondfs to different hyperparalmeters for your specific dataset.
    # Don't heistate to click on "explosion" if the embedding seems stuck, and to increase the learning rate on the top.
    N, M, X, Y = fetch_MNIST(atrificially_inflate_n_times=15) # 15 * 60k = 900k points
    Xld, _ = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y,\
                base_attraction_repulsion_ratio=80.0, end_attrac_mult=0.9, end_kernel_alpha=80.0, lr_strength=15.0)

    # 5. example usage for a dataset with the properties of "blob": need to do high attraction and high LD kernel alpha (ie: low tails: closer to SNE that the Student-t verions)
    # Often, if you want to exxagerate the separation between the clusters, you will want to exxagerate the by-design discrepancy between the kernel tails in HD (gaussian, adaptive to the local density), and those in LD (no local adaptaion, bu tunable tail heaviness);
    # having heavier tails (low "kernel alpha") would encourage h-t-SNE to tear the manifold in weak zones (low density) and exxagerate the sepâration of distinct cluster, facilitating visualisation.
    # However, on some rare datasets such as blobs, having heavy tails in LD (low "kernel alpha") gives the opposite result: the intra-cluster repulsions make the clusters spread out and touch other clusters at the boundary.
    # it's imporetant to play with the "kernel alpha" hyperparameter to see where and how easily the HD structures get torn appart by h-t-SNE.
    N, M, X, Y = get_blobs(n=300*1000)
    Xld, knn_HD = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y, end_attrac_mult=0.8, end_kernel_alpha=80.0)
    
    # 6. example usage if the aim is to get the KNN sets of the data instead of the embedding
    # the embedding is built slower, but the KNN sets are built faster (exit the gui with the escape key when satisfied)
    # printed in the terminal is a value that vaguely correlates with the KNN error: leaving the optimsiation around 0.02
    # gives fast and pretty good knns, waiiting for 0.00 gives really nice knn quality, for a bit more time to compute.
    # clocking a couple of times on "reset" or "explosion" after the warmup period can help shake out the last drops of error.
    N, M, X, Y = fetch_MNIST(atrificially_inflate_n_times=1)
    Xld, knn_HD = htSNE(n_components=2, verbose=True, with_gui=True).fit(X, Y, purpose_is_KNN = True)

    # 7. example of how to read a saved embedding and plot it with matplotlib
    """ xld = np.load("Xld_PP=80.0_KA=70.0_AM=0.9_DM=0" + ".npy")
    y = np.load("Y_PP=80.0_KA=70.0_AM=0.9_DM=0" + ".npy")
    import matplotlib.pyplot as plt
    plt.scatter(xld[:, 0], xld[:, 1], c=y, s=0.1)
    plt.show() """

if __name__ == "__main__":
    run_demo()
