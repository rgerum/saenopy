import numpy as np
import scipy.fft
from scipy.ndimage import median_filter, gaussian_filter
from scipy.ndimage import uniform_filter

from .suppress_warnings import suppress_warnings


def ffttc_traction(
    u,
    v,
    pixel_size1,
    pixel_size2,
    young,
    sigma=0.49,
    spatial_filter="gaussian",
    fs=None,
):
    """
    fourier transform based calculation of the traction force. U and v must be given  as deformations in pixel. Size of
    these pixels must be the pixel size (size of a pixel in the deformation field u or v). Note that thePiv deformation
    returns deformation in pixel of the size of pixels in the images of beads before and after.
    If bf_image is provided this script will return a traction field that is zoomed to the size
     of the bright field image, by interpolation. It is not recommended to use this for any calculations.
    The function can use different filters. Recommended filter is gaussian. Mean filter should yield similar results.

    :param u:deformation field in x direction in pixel of the deformation image
    :param v:deformation field in y direction in pixel of the deformation image
    :param young: Young's modulus in Pa
    :param pixel_size1: pixel size in m/pixel of the original image, needed because u and v is given as
    displacement of these pixels
    :param pixel_size2: pixel size of m/pixel the deformation image
    :param sigma: poisson ratio of the gel
    :param spatial_filter: str, values: "mean","gaussian","median". Different smoothing methods for the traction field
    :return: tx_filter,ty_filter: traction forces in x and y direction in Pa

    Parameters
    ----------
    fs
    """

    # 0) subtracting mean(better name for this step)
    u_shift = u - np.mean(
        u
    )  # shifting to zero mean  (translating to pixel size of u-image is done later)
    v_shift = v - np.mean(v)

    # Ben's algorithm:
    # 1)Zero padding to get square array with even index number
    ax1_length = np.shape(u_shift)[0]  # u and v must have same dimensions
    ax2_length = np.shape(u_shift)[1]
    max_ind = int(np.max((ax1_length, ax2_length)))
    if max_ind % 2 != 0:
        max_ind += 1

    u_expand = np.zeros((max_ind, max_ind))
    v_expand = np.zeros((max_ind, max_ind))
    u_expand[:ax1_length, :ax2_length] = u_shift
    v_expand[:ax1_length, :ax2_length] = v_shift

    # 2) producing wave vectors
    # form 1:max_ind/2 then -(max_ind/2:1)
    kx1 = np.array(
        [
            list(range(0, int(max_ind / 2), 1)),
        ]
        * int(max_ind)
    )
    kx2 = np.array(
        [
            list(range(-int(max_ind / 2), 0, 1)),
        ]
        * int(max_ind)
    )
    kx = (
        np.append(kx1, kx2, axis=1) * 2 * np.pi
    )  # fourier transform in this case is defined as
    # F(kx)=1/2pi integral(exp(i*kx*x)dk therefore kx must be expressed as a spatial frequency in distance*2*pi
    ky = np.transpose(kx)
    k = np.sqrt(kx**2 + ky**2) / (pixel_size2 * max_ind)

    # 2.1) calculating angle between k and kx with atan2 function
    alpha = np.arctan2(ky, kx)
    alpha[0, 0] = np.pi / 2

    # 3) calculation of K --> Tensor to calculate displacements from traction forces. We calculate inverse of K
    # K⁻¹=[[kix kid],
    #     [kid,kiy]]  ,,, so is "diagonal, kid appears two times
    kix = ((k * young) / (2 * (1 - sigma**2))) * (
        1 - sigma + sigma * np.cos(alpha) ** 2
    )
    kiy = ((k * young) / (2 * (1 - sigma**2))) * (
        1 - sigma + sigma * np.sin(alpha) ** 2
    )
    kid = ((k * young) / (2 * (1 - sigma**2))) * (sigma * np.sin(alpha) * np.cos(alpha))

    # adding zeros in kid diagonals
    kid[:, int(max_ind / 2)] = np.zeros(max_ind)
    kid[int(max_ind / 2), :] = np.zeros(max_ind)

    # 4) calculate Fourier transform of displacement
    # u_ft=np.fft.fft2(u_expand*pixel_size1*2*np.pi)
    # v_ft=np.fft.fft2(v_expand*pixel_size1*2*np.pi)
    u_ft = scipy.fft.fft2(u_expand * pixel_size1)
    v_ft = scipy.fft.fft2(v_expand * pixel_size1)

    # 4.1) calculate traction forces in Fourier space T=K⁻¹*U, U=[u,v] here with individual matrix elements.
    tx_ft = kix * u_ft + kid * v_ft
    ty_ft = kid * u_ft + kiy * v_ft

    # 4.2) go back to real space
    tx = scipy.fft.ifft2(tx_ft).real
    ty = scipy.fft.ifft2(ty_ft).real

    # 5.2) cut back to original shape
    tx_cut = tx[0:ax1_length, 0:ax2_length]
    ty_cut = ty[0:ax1_length, 0:ax2_length]

    # 5.3) using filter
    tx_filter = tx_cut
    ty_filter = ty_cut

    if spatial_filter == "mean":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(int(np.max((ax1_length, ax2_length))) / 16)
        )
        tx_filter = uniform_filter(tx_cut, size=fs)
        ty_filter = uniform_filter(ty_cut, size=fs)
    if spatial_filter == "gaussian":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(np.max((ax1_length, ax2_length))) / 50
        )
        tx_filter = gaussian_filter(tx_cut, sigma=fs)
        ty_filter = gaussian_filter(ty_cut, sigma=fs)
    if spatial_filter == "median":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(int(np.max((ax1_length, ax2_length))) / 16)
        )
        tx_filter = median_filter(tx_cut, size=fs)
        ty_filter = median_filter(ty_cut, size=fs)

    return tx_filter, ty_filter


def ffttc_traction_finite_thickness(
    u,
    v,
    pixel_size1,
    pixel_size2,
    h,
    young,
    sigma=0.49,
    spatial_filter="gaussian",
    fs=None,
):
    """
    FTTC with correction for finite substrate thickness according to
    Xavier Trepat, Physical forces during collective cell migration, 2009

    :param u:deformation field in x direction in pixel of the deformation image
    :param v:deformation field in y direction in pixel of the deformation image
    :param young: Young's modulus in Pa
    :param pixel_size1: pixel size of the original image, needed because u and v is given as displacement of these pixels
    :param pixel_size2: pixel size of the deformation image
    :param h: height of the membrane the cells lie on, in µm
    :param sigma: Poisson's ratio of the gel
    :param spatial_filter: str, values: "mean","gaussian","median". Different smoothing methods for the traction field.
    :param fs: float, size of the filter (std of gaussian or size of the filter window) in µm
    :return: tx_filter,ty_filter: traction forces in x and y direction in Pa
    """

    # 0) subtracting mean(better name for this step)
    u_shift = (u - np.mean(u)) * pixel_size1
    v_shift = (v - np.mean(v)) * pixel_size1

    # Ben's algorithm:
    # 1)Zero padding to get square array with even index number
    ax1_length = np.shape(u_shift)[0]  # u and v must have same dimensions
    ax2_length = np.shape(u_shift)[1]
    max_ind = int(np.max((ax1_length, ax2_length)))
    if max_ind % 2 != 0:
        max_ind += 1
    u_expand = np.zeros((max_ind, max_ind))
    v_expand = np.zeros((max_ind, max_ind))
    u_expand[max_ind - ax1_length : max_ind, max_ind - ax2_length : max_ind] = u_shift
    v_expand[max_ind - ax1_length : max_ind, max_ind - ax2_length : max_ind] = v_shift

    # 2) producing wave vectors (FT-space)
    # form 1:max_ind/2 then -(max_ind/2:1)
    kx1 = np.array(
        [
            list(range(0, int(max_ind / 2), 1)),
        ]
        * int(max_ind),
        dtype=np.float64,
    )
    kx2 = np.array(
        [
            list(range(-int(max_ind / 2), 0, 1)),
        ]
        * int(max_ind),
        dtype=np.float64,
    )
    # spatial frequencies: 1/wavelength,in 1/µm in fractions of total length

    kx = np.append(kx1, kx2, axis=1) * 2 * np.pi / (pixel_size2 * max_ind)
    ky = np.transpose(kx)
    k = np.sqrt(kx**2 + ky**2)  # matrix with "relative" distances??#

    r = k * h
    c = np.cosh(r)
    s = np.sinh(r)
    s_c = np.tanh(r)

    # gamma = ((3 - 4 * sigma) * (c ** 2) + (1 - 2 * sigma) ** 2 + (k * h) ** 2) / (
    #         (3 - 4 * sigma) * s * c + k * h)  ## inf values here because k goes to zero
    gamma = (
        (3 - 4 * sigma) + (((1 - 2 * sigma) ** 2) / (c**2)) + ((r**2) / (c**2))
    ) / ((3 - 4 * sigma) * s_c + r / (c**2))

    # 4) calculate fourier transform of displacements
    u_ft = scipy.fft.fft2(u_expand)
    v_ft = scipy.fft.fft2(v_expand)

    """
    #4.0*) approximation for large h according to this paper
    factor3=young/(2*(1-sigma**2)*k)
    factor3[0,0]=factor3[0,1]
    tx_ft=factor3*(u_ft*((k**2)*(1-sigma)+sigma*(kx**2)) + v_ft*kx*ky*sigma)
    ty_ft=factor3*(v_ft*((k**2)*(1-sigma)+sigma*(ky**2)) + u_ft*kx*ky*sigma)
    """

    # 4.1) calculate traction forces in Fourier space
    factor1 = v_ft * kx - u_ft * ky
    factor2 = u_ft * kx + v_ft * ky
    tx_ft = ((-young * ky * c) / (2 * k * s * (1 + sigma))) * factor1 + (
        (young * kx) / (2 * k * (1 - sigma**2))
    ) * gamma * factor2
    tx_ft[0, 0] = 0  # zero frequency would represent force everywhere (constant)
    ty_ft = ((young * kx * c) / (2 * k * s * (1 + sigma))) * factor1 + (
        (young * ky) / (2 * k * (1 - sigma**2))
    ) * gamma * factor2
    ty_ft[0, 0] = 0

    # 4.2) go back to real space
    tx = scipy.fft.ifft2(tx_ft.astype(np.complex128)).real
    ty = scipy.fft.ifft2(ty_ft.astype(np.complex128)).real

    # 5.2) cut like in script from ben
    tx_cut = tx[max_ind - ax1_length : max_ind, max_ind - ax2_length : max_ind]
    ty_cut = ty[max_ind - ax1_length : max_ind, max_ind - ax2_length : max_ind]

    # 5.3) using filter
    tx_filter = tx_cut
    ty_filter = ty_cut
    if spatial_filter == "mean":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(int(np.max((ax1_length, ax2_length))) / 16)
        )
        tx_filter = uniform_filter(tx_cut, size=fs)
        ty_filter = uniform_filter(ty_cut, size=fs)
    if spatial_filter == "gaussian":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(np.max((ax1_length, ax2_length))) / 50
        )
        tx_filter = gaussian_filter(tx_cut, sigma=fs)
        ty_filter = gaussian_filter(ty_cut, sigma=fs)
    if spatial_filter == "median":
        fs = (
            fs
            if isinstance(fs, (float, int))
            else int(int(np.max((ax1_length, ax2_length))) / 16)
        )
        tx_filter = median_filter(tx_cut, size=fs)
        ty_filter = median_filter(ty_cut, size=fs)

    return tx_filter, ty_filter


def TFM_tractions(
    u,
    v,
    pixel_size1,
    pixel_size2,
    h,
    young,
    sigma=0.49,
    spatial_filter="gaussian",
    fs=6,
):
    """
    height correction breaks down due to numerical reasons at large gel height and small wavelengths of deformations.
    In this case the height corrected ffttc-function returns Nans. This function falls back
    to the non height-corrected FTTC function if this happens
    :return:
    """
    # translate the filter size to pixels of traction field
    fs = fs / pixel_size2 if isinstance(fs, (int, float)) else None
    if isinstance(h, (int, float)):
        with suppress_warnings(RuntimeWarning):
            tx, ty = ffttc_traction_finite_thickness(
                u,
                v,
                pixel_size1=pixel_size1,
                pixel_size2=pixel_size2,
                h=h,
                young=young,
                sigma=sigma,
                spatial_filter=spatial_filter,
                fs=fs,
            )  # unit is N/m**2
            if np.any(np.isnan(tx)) or np.any(np.isnan(ty)):
                tx, ty = ffttc_traction(
                    u,
                    v,
                    pixel_size1=pixel_size1,
                    pixel_size2=pixel_size2,
                    young=young,
                    sigma=sigma,
                    spatial_filter=spatial_filter,
                    fs=fs,
                )

    elif h == "infinite":
        tx, ty = ffttc_traction(
            u,
            v,
            pixel_size1=pixel_size1,
            pixel_size2=pixel_size2,
            young=young,
            sigma=sigma,
            spatial_filter=spatial_filter,
            fs=fs,
        )
    else:
        raise ValueError("illegal value for h")
    return tx, ty
