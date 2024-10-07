#include the classics
import numpy as np
import matplotlib.pyplot as plt

#include both modules
import fastSNE.fastSNE as fastSNE

# set the random seed
# np.random.seed(42)

def fetch_dataset():

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
    n_times_larger = 1
    N_new = N * n_times_larger
    X_new = np.zeros((N_new, M), dtype=np.float32)
    Y_new = np.zeros((N_new, 1), dtype=np.float32)
    perms = np.random.permutation(N)
    for i in range(n_times_larger):
        X_new[i*N:(i+1)*N, :] = X[perms, :]
        Y_new[i*N:(i+1)*N, :] = Y[perms, :]

    # shuffle the dataset
    perms = np.random.permutation(N_new)
    X_new = X_new[perms, :]
    Y_new = Y_new[perms, :]

    # add some noise
    X_new += np.random.normal(0, 5.0, X_new.shape)

    return N_new, M, X_new, Y_new.astype(np.int32)

# now do explode button and hirizontal lr mult 


"""
nouveau truc : 
def get_coil20(N, dataset_params):
    from scipy.io import loadmat
    mat = loadmat("datasets/COIL20.mat")
    X, Y = mat['X'], mat['Y']
    Y = (Y.astype(int) - 1).reshape((-1,))
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None
    qsdqd
    # dataset_params['supervised feature selection'] = True


nouveau aussi
def get_RNAseq(N, dataset_params):
    if N == 3000:
        filename = 'datasets/RNAseq_N3k.npy'
    elif N == 6000:
        filename = 'datasets/RNAseq_N10k.npy'
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

    dataset_params['X'] = X[:N]
    dataset_params['Y'] = Y[:N]
    dataset_params['is_classification'] = True
    dataset_params['colors'] = rgb_colors

"""


def run_demo():
    # fetch the dataset
    N, M, X, Y = fetch_dataset()

    with_GUI = True 
    tsne = fastSNE.fastSNE(with_GUI, n_components=2, random_state=None)
    if with_GUI:
        tsne.fit(N, M, X, Y)
    else:
        tsne.fit(N, M, X, Y=None)
    Xld = tsne.transform()

    """ # save the results
    plt.scatter(Xld[:, 0], Xld[:, 1])
    plt.savefig('tsne_plot.png')  """

    return 42

if __name__ == '__main__':
    run_demo()
    print("program ended")