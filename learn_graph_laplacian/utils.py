import numpy as np
from scipy import sparse
import time
from scipy import spatial
from pyunlocbox import functions, solvers

def compute_A(X, dist_type='sqeuclidean', alpha=1, s=None, step=0.5,
                  w0=None, maxit=1000, rtol=1e-5, retall=False,
                  verbosity='NONE') :
    
    # Parse X
    N = X.shape[0]
    E = int(N * (N - 1.) / 2.)
    z = spatial.distance.pdist(X, dist_type)  # Pairwise distances

    # Parse s
    s = N if s is None else s

    # Parse step
    if (step <= 0) or (step > 1):
        raise ValueError("step must be a number between 0 and 1.")

    # Parse initial weights
    w0 = np.zeros(z.shape) if w0 is None else w0
    if (w0.shape != z.shape):
        raise ValueError("w0 must be of dimension N(N-1)/2.")

    # Get primal-dual linear map
    one_vec = np.ones(E)

    def K(w):
        return np.array([2 * np.dot(one_vec, w)])

    def Kt(n):
        return 2 * n * one_vec

    norm_K = 2 * np.sqrt(E)

    # Get weight-to-degree map
    S, St = weight2degmap(N)

    # Assemble functions in the objective
    f1 = functions.func()
    f1._eval = lambda w: 2 * np.dot(w, z)
    f1._prox = lambda w, gamma: np.maximum(0, w - (2 * gamma * z))

    f2 = functions.func()
    f2._eval = lambda w: 0.
    f2._prox = lambda d, gamma: s

    f3 = functions.func()
    f3._eval = lambda w: alpha * (2 * np.sum(w**2) + np.sum(S(w)**2))
    f3._grad = lambda w: alpha * (4 * w + St(S(w)))
    lipg = 2 * alpha * (N + 1)

    # Rescale stepsize
    stepsize = step / (1 + lipg + norm_K)

    # Solve problem
    solver = solvers.mlfbf(L=K, Lt=Kt, step=stepsize)
    problem = solvers.solve([f1, f2, f3], x0=w0, solver=solver, maxit=maxit,
                            rtol=rtol, verbosity=verbosity)

    # Transform weight matrix from vector form to matrix form
    W = spatial.distance.squareform(problem['sol'])

    if retall:
        return W, problem
    else:
        return W

def weight2degmap(N, array=False) :

    Ne = int(N * (N - 1) / 2)  # Number of edges
    row_idx1 = np.zeros((Ne, ))
    row_idx2 = np.zeros((Ne, ))
    count = 0
    for i in np.arange(1, N):
        row_idx1[count: (count + (N - i))] = i - 1
        row_idx2[count: (count + (N - i))] = np.arange(i, N)
        count = count + N - i
    row_idx = np.concatenate((row_idx1, row_idx2))
    col_idx = np.concatenate((np.arange(0, Ne), np.arange(0, Ne)))
    vals = np.ones(len(row_idx))
    K = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(N, Ne))
    if array:
        return K, K.transpose()
    else:
        return lambda w: K.dot(w), lambda d: K.transpose().dot(d)


