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