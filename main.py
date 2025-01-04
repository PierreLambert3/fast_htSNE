#include the classics
import numpy as np
import matplotlib.pyplot as plt

#include both modules
import fastSNE.fastSNE as fastSNE

# set the random seed
# np.random.seed(42)

def fetch_MNIST(atrificially_inflate_n_times=1):

    # binary dump from a C flattened array of floats (32bits)
    mnist_binaries_path = r"C:\Users\pierr\dev\datasets\MNIST_50PC"
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
    X, Y = make_blobs(n_samples= 20000, n_features=64, centers=18, cluster_std=6.0, center_box=(-10.0, 10.0), shuffle=True)
    N, M = X.shape
    return N, M, X.astype(np.float32), Y.astype(np.int32)

import time
def umap_embedding(X, Y):
    import umap.umap_ as umap
    start = time.time()
    reducer = umap.UMAP(n_components=2)
    reducer.fit(X)
    embedding = reducer.transform(X)
    print("UMAP took ", time.time() - start, " seconds")
    neighbours = reducer.nearest_neighbors_.idx
    return embedding, neighbours

def get_umap_neighbours(X, k):
    # umap.umap_.nearest_neighbors(X, n_neighbors, metric, metric_kwds, angular, random_state, low_memory=True, use_pynndescent=True, n_jobs=-1, verbose=False)
    import umap.umap_ as umap
    start = time.time()
    knn = umap.nearest_neighbors(X, k+1, metric='euclidean', metric_kwds={}, angular=False, random_state=None)[0][:, 1:]
    time_taken = time.time() - start
    return knn, time_taken

def sort_neighbors_by_distance(X, approx_knn):
    """Sort approximate neighbors by distance for each point"""
    import numpy as np
    N, k = approx_knn.shape
    
    # Calculate distances to neighbors
    distances = np.zeros((N, k))
    for i in range(N):
        point = X[i]
        neighbors = X[approx_knn[i]]
        distances[i] = np.sum((neighbors - point[None, :]) ** 2, axis=1)
    
    # Sort neighbors by distance
    sort_idx = np.argsort(distances, axis=1)
    sorted_knn = np.zeros_like(approx_knn)
    for i in range(N):
        sorted_knn[i] = approx_knn[i][sort_idx[i]]
    
    return sorted_knn

def evaluate_neighbors_quality_fast(X, approx_knn_umap, approx_knn_mine, k, n_samples=10000):
    approx_knn_umap = sort_neighbors_by_distance(X, approx_knn_umap)
    approx_knn_umap = approx_knn_umap[:, :k]
    approx_knn_mine = sort_neighbors_by_distance(X, approx_knn_mine)
    approx_knn_mine = approx_knn_mine[:, :k]
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    
    # 1. Sample points
    n_total = X.shape[0]
    sample_idx = np.random.choice(n_total, size=n_samples, replace=False)
    
    # 2. Get ground truth for sampled points (k+1 to account for self)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(X)
    _, true_indices = nbrs.kneighbors(X[sample_idx])
    true_indices = true_indices[:, 1:]
    
    # 3. Get UMAP neighbors for sampled points
    umap_indices = approx_knn_umap[sample_idx]
    mine_indices = approx_knn_mine[sample_idx]
    
    # 4-7. Compute intersection scores at each scale
    scale_scores_umap = []
    scale_scores_mine = []
    for h in range(1, k):
        intersections_umap = []
        intersections_mine = []
        for i in range(n_samples):
            true_set = set(true_indices[i, :h])
            umap_set = set(umap_indices[i, :h])
            overlap = len(true_set.intersection(umap_set))
            intersections_umap.append(overlap / h)  # normalize by scale
            mine_set = set(mine_indices[i, :h])
            overlap = len(true_set.intersection(mine_set))
            intersections_mine.append(overlap / h)
            
        scale_scores_umap.append(np.mean(intersections_umap))
        scale_scores_mine.append(np.mean(intersections_mine))
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, k), scale_scores_umap, '-', label='UMAP', color='red')
    plt.plot(range(1, k), scale_scores_mine, '-', label='Mine', color='blue')
    plt.xlabel('Scale (h)')
    plt.ylabel('Mean Intersection Score')
    plt.title('KNN Quality vs Scale')
    plt.grid(True)
    plt.legend()
    plt.show()
    """ plt.plot(range(1, k), scale_scores, '-o')
    plt.xlabel('Scale (h)')
    plt.ylabel('Mean Intersection Score')
    plt.title('KNN Quality vs Scale')
    plt.grid(True)
    plt.show() """
    
    # return scale_scores

def run_demo():
    #  The 60k train set of MNIST, reduced to 50 dimensions with PCA
    inflate_n_times = 1 # if > 1 : creates new observations by copying the original ones and adding noise
    
    N, M, X, Y = fetch_MNIST(inflate_n_times) #  X.shape = (inflate_n_times*60k, 50)
    # N, M, X, Y = get_RNAseq()
    # N, M, X, Y = get_coil20()
    # N, M, X, Y = get_blobs()


    k = 256

    # umap_Xld, umap_KNN = umap_embedding(X, Y)
    """ k = 15
    umap_KNN = get_umap_neighbours(X, k+1)
    # umap_KNN = np.random.randint(0, N, (N, k))
    print(umap_KNN)
    print()
    scores = evaluate_neighbors_quality_fast(X, umap_KNN, n_samples=10000)
    print(scores) """

    """ 
    1/0

    # resultats a montrer : 
    # SNE, tSNE PP=5, tSNE PP=60, UMAP, htSNE

    print("N = ", N, " M = ", M) """

    tsne = fastSNE.fastSNE(n_components=2, random_state=None)
    # knn_umap, time_taken = get_umap_neighbours(X, k)
    time_taken = None
    Xld, knn_mine = tsne.fit(N, M, X, Y, max_n_sec=time_taken).transform()
    # scores = evaluate_neighbors_quality_fast(X, knn_umap, knn_mine, k, n_samples=5000)
    scores = evaluate_neighbors_quality_fast(X, knn_mine, knn_mine, k, n_samples=5000)

    """ from matplotlib import pyplot as plt
    plt.scatter(Xld[:, 0], Xld[:, 1], c=Y, s=0.4)
    plt.show() """

    return 42

if __name__ == '__main__':
    run_demo()