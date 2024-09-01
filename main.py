#include the classics
import numpy as np
import matplotlib.pyplot as plt

#include both modules
import fastSNE.fastSNE as fastSNE

def fetch_dataset():
    # binary dump from a C flattened array of floats (32bits)
    mnist_binaries_path = "/home/gneeeeeh/dev/datasets/bin/mnist/"
    mnist_X_filename = mnist_binaries_path + "MNIST_PCA_X.bin"
    mnist_Y_filename = mnist_binaries_path + "MNIST_PCA_Y.bin"
    N = 60*1000
    M = 50

    # load the dataset
    X = np.fromfile(mnist_X_filename, dtype=np.float32).reshape(N, M)
    Y = np.fromfile(mnist_Y_filename, dtype=np.float32).reshape(N, 1)

    # allocate new dataset
    n_times_larger = 7
    N_new = 60*1000 * n_times_larger
    X_new = np.zeros((N_new, M), dtype=np.float32)
    Y_new = np.zeros((N_new, 1), dtype=np.float32)

    for i in range(n_times_larger):
        X_new[i*N:(i+1)*N, :] = X[np.random.permutation(N), :]
        Y_new[i*N:(i+1)*N, :] = Y[np.random.permutation(N), :]

    # shuffle the dataset
    perms = np.random.permutation(N_new)
    X_new = X_new[perms, :]
    Y_new = Y_new[perms, :]

    # add some noise
    X_new += np.random.normal(0, 0.1, X_new.shape)

    return N_new, M, X_new, Y_new.astype(np.int32)
    

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