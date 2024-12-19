#include the classics
import numpy as np
import matplotlib.pyplot as plt

#include both modules
import fastSNE.fastSNE as fastSNE

# set the random seed
# np.random.seed(42)

def fetch_MNIST(atrificially_inflate_n_times=1):

    # binary dump from a C flattened array of floats (32bits)
    mnist_binaries_path = r"C:\Users\pierr\dev\datasets\bin\mnist"
    mnist_X_filename = mnist_binaries_path + r"\MNIST_PCA_X.bin"
    mnist_Y_filename = mnist_binaries_path + r"\MNIST_PCA_Y.bin"
    N = 60*1000
    M = 50

    # load the dataset
    X = np.fromfile(mnist_X_filename, dtype=np.float32).reshape(N, M)
    Y = np.fromfile(mnist_Y_filename, dtype=np.float32).reshape(N, 1)


    # N_sampled = 24000
    # perms = np.random.permutation(N)
    # X = X[perms[:N_sampled], :]
    # Y = Y[perms[:N_sampled], :]
    # N = N_sampled 
    # return N, M, X, Y.astype(np.int32)

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

    return N_new, M, X_new, Y_new.astype(np.int32)

def load_zfish_timeLabels():
    X = np.load("datasets/zfish/zfish_X.npy")
    N, M = X.shape
    colours = np.load("datasets/zfish/zfish_stageRGB.npy")
    from sklearn.decomposition import PCA
    M = 20
    X = PCA(n_components=M).fit_transform(X)
    return N, M, X.astype(np.float32), colours / 256.0

def load_zfish_classif():
    X = np.load("datasets/zfish/zfish_X.npy").astype(np.float32)
    N, M = X.shape
    colours = np.load("datasets/zfish/zfish_classes.npy")
    # pca of X
    from sklearn.decomposition import PCA
    M = 20
    X = PCA(n_components=M).fit_transform(X)
    return N, M, X.astype(np.float32), colours

def get_RNAseq():
    filename = 'datasets/RNAseq_N20k.npy'
    XY = np.load(filename)
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

def get_coil20():
    from scipy.io import loadmat
    mat = loadmat("datasets/COIL20.mat")
    X, Y = mat['X'], mat['Y']
    Y = (Y.astype(int) - 1).reshape((-1,))
    N, M = X.shape
    return N, M, X.astype(np.float32), Y.astype(np.int32)

def get_blobs():
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples= 300 * 1000, n_features=15, centers=9, cluster_std=6.0)
    N, M = X.shape
    return N, M, X.astype(np.float32), Y

def run_demo():
    #  The 60k train set of MNIST, reduced to 50 dimensions with PCA
    inflate_n_times = 1 # if > 1 : creates new observations by copying the original ones and adding noise
    
    N, M, X, Y = fetch_MNIST(inflate_n_times) #  X.shape = (inflate_n_times*60k, 50)
    # N, M, X, Y = get_RNAseq()
    # N, M, X, Y = get_coil20()

    print("N = ", N, " M = ", M)

    tsne = fastSNE.fastSNE(n_components=2, random_state=None)
    Xld = tsne.fit(N, M, X, Y).transform()

    return 42

if __name__ == '__main__':
    run_demo()