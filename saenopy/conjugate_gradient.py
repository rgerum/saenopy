import numpy as np


def cg(A: np.ndarray, b: np.ndarray, maxiter: int = 1000, tol: float = 0.00001, verbose: bool = False):
    """ solve the equation Ax=b with the conjugate gradient method """
    def norm(x):
        return np.inner(x.flatten(), x.flatten())

    # calculate the total force "amplitude"
    normb = norm(b)

    # if it is not 0 (always has to be positive)
    if normb == 0:
        return 0

    x = np.zeros_like(b)

    # the difference between the desired force deviations and the current force deviations
    r = b - A @ x

    # and store it also in pp
    p = r

    # calculate the total force deviation "amplitude"
    resid = norm(p)

    # iterate maxiter iterations
    for i in range(1, maxiter + 1):
        Ap = A @ p

        alpha = resid / np.sum(p * Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = norm(r)

        # check if we are already below the convergence tolerance
        if rsnew < tol * normb:
            break

        beta = rsnew / resid

        # update pp and resid
        p = r + beta * p
        resid = rsnew

        # print status every 100 frames
        if i % 100 == 0 and verbose:
            print(i, ":", resid, "alpha=", alpha, "du=", np.sum(x ** 2))  # , end="\r")

    return x
