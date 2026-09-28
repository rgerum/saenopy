import numpy as np


def cg(A: np.ndarray, b: np.ndarray, maxiter: int = 1000, tol: float = 0.00001,
       verbose: bool = False, *, return_info=False, cancel_signal=None):
    """Solve Ax=b. ``tol`` is the SQUARED relative residual tolerance.

    Optional diagnostics report the explicitly recomputed residual, including
    iteration-limit exits. The default return remains the solution vector.
    """
    if not np.isfinite(tol) or tol <= 0 or int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("CG requires a positive finite tolerance and positive integer maxiter")
    maxiter = int(maxiter)
    def norm(x):
        return np.inner(x.flatten(), x.flatten())

    # calculate the total force "amplitude"
    normb = norm(b)
    if not np.isfinite(normb):
        raise ValueError("CG right-hand side contains nonfinite values")

    x = np.zeros_like(b)
    def finish(iterations, reason):
        relative_residual = float(np.sqrt(norm(b - A @ x) / normb)) if normb else 0.0
        converged = bool(np.isfinite(relative_residual) and relative_residual <= np.sqrt(tol))
        info = dict(iterations=iterations, maxiter=int(maxiter),
                    relative_residual=relative_residual, relative_tolerance=float(np.sqrt(tol)),
                    converged=converged, reason="converged" if converged else reason)
        return (x, info) if return_info else x

    # if it is not 0 (always has to be positive)
    if normb == 0:
        return finish(0, "converged")

    # the difference between the desired force deviations and the current force deviations
    r = b - A @ x

    # and store it also in pp
    p = r

    # calculate the total force deviation "amplitude"
    resid = norm(p)

    # iterate maxiter iterations
    for i in range(1, maxiter + 1):
        if cancel_signal is not None and getattr(cancel_signal, "cancel", False):
            return finish(i - 1, "cancelled")
        Ap = A @ p

        curvature = np.sum(p * Ap)
        if not np.isfinite(curvature) or curvature == 0:
            raise FloatingPointError("CG breakdown: zero or nonfinite search-direction curvature")
        alpha = resid / curvature

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = norm(r)
        if not np.isfinite(rsnew):
            raise FloatingPointError("CG residual became nonfinite")

        # check if we are already below the convergence tolerance
        if rsnew <= tol * normb:
            # Recursive residuals can drift in ill-conditioned systems. Verify
            # Ax=b before declaring convergence; restart if accuracy was lost.
            r = b - A @ x
            rsnew = norm(r)
            if rsnew <= tol * normb:
                return finish(i, "converged")
            p = r.copy()
            resid = rsnew
            continue

        beta = rsnew / resid

        # update pp and resid
        p = r + beta * p
        resid = rsnew

        # print status every 100 frames
        if i % 100 == 0 and verbose:
            print(i, ":", resid, "alpha=", alpha, "du=", np.sum(x ** 2))  # , end="\r")

    return finish(int(maxiter), "iteration_limit")
