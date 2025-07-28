import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact
import random
import scipy
from copy import deepcopy as cp
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import os
from pathlib import Path
from .utils import repeat
from scipy.signal import find_peaks, peak_widths

def gaussian_2D(x, pref, mean, std):
    if isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, type(np.array([1]))):
        raise TypeError('You must provide a list or a numpy array!')
    return pref * (2*np.pi*std**2)**(-0.5) * np.exp(-(x-mean)**2/(2*std**2))

def line_2D(x, a, q):
    return a*x + q

def lorentzian_2D(x, area, center, gamma):
    '''
    Lorentzian with gamma = FULL width at half maximum
    '''
    if isinstance(x, list):
        x = np.array(x)
    elif not isinstance(x, type(np.array([1]))):
        raise TypeError('You must provide a list or a numpy array!')
    return area * 1/(np.pi) * (gamma/2) / ((x-center)**2 + (gamma/2)**2)

def find_indices(values, array):
    """
    Find indices in `array` corresponding to the closest values to the given `values`.

    For each element in `values`, this function finds the index in `array` where the
    absolute difference is minimal.

    Parameters
    ----------
    values : array_like
        A list or array of values to search for in `array`.
    array : array_like
        The array in which to find the closest match to each value in `values`.

    Returns
    -------
    inds : ndarray of int
        Array of indices such that `array[inds[i]]` is the closest value in `array` to `values[i]`.

    Notes
    -----
    Both `values` and `array` are internally converted to NumPy arrays (via deep copy) to ensure safe computation.

    Examples
    --------
    >>> find_indices([2.1, 4.8], [1.0, 2.0, 3.0, 5.0])
    array([1, 3])
    """
    values = np.array(cp(values))
    array = np.array(cp(array))
    inds = []
    for value in values:
        inds.append(np.argmin(np.abs(array-value)))
    inds = np.array(inds)
    return inds
            

def sum_operation(args):
    return sum(args)
    
def boh(args):
    return args[0] + 3* (args[1])

def fwhm_finder(x, y, x_peak=None, plot=False):
    """
    Estimate the Full Width at Half Maximum (FWHM) of a single peak in a spectrum.

    Parameters
    ----------
    x : array_like
        Array of x values (e.g., frequency or energy).
    y : array_like
        Array of y values (e.g., intensity).
    x_peak : float, optional
        x position of the peak. If None, the global maximum is used.
    plot : bool
        If True, plot the spectrum and mark the FWHM range.

    Returns
    -------
    fwhm : float
        Full Width at Half Maximum.
    x_l : float
        x position of the left half-maximum.
    x_r : float
        x position of the right half-maximum.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if x_peak is None:
        idx_peak = np.argmax(y)
    else:
        idx_peak = np.argmin(np.abs(x - x_peak))

    y_half = y[idx_peak] / 2.0

    # Search to the left
    i_left = np.where(y[:idx_peak] < y_half)[0]
    if len(i_left) == 0:
        raise ValueError("No left crossing found.")
    i_l = i_left[-1]
    x_l = np.interp(y_half, [y[i_l], y[i_l + 1]], [x[i_l], x[i_l + 1]])

    # Search to the right
    i_right = np.where(y[idx_peak + 1:] < y_half)[0]
    if len(i_right) == 0:
        raise ValueError("No right crossing found.")
    i_r = i_right[0] + idx_peak + 1
    x_r = np.interp(y_half, [y[i_r - 1], y[i_r]], [x[i_r - 1], x[i_r]])

    fwhm = x_r - x_l

    if plot:
        plt.plot(x, y, label="Spectrum")
        plt.axvline(x_l, color='red', linestyle='--', label="Half max bounds")
        plt.axvline(x_r, color='red', linestyle='--')
        plt.axhline(y_half, color='gray', linestyle=':')
        plt.scatter([x[idx_peak]], [y[idx_peak]], color='black', label="Peak")
        plt.title(f"FWHM = {fwhm:.4f}")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()

    return fwhm, i_l, i_r

def peak_finder(x, ref_yy, n_peaks=None):
    """
    Identify peaks in the signal and return their indices, x, and y values.

    Parameters
    ----------
    x : array_like
        The x-values of the data.
    ref_yy : array_like
        The y-values of the signal.
    n_peaks : int or None
        Number of top peaks to return, sorted by peak height. If None, return all peaks.

    Returns
    -------
    ndarray of shape (N, 3)
        Each row contains [index, x[index], ref_yy[index]], sorted by increasing energy.
    """
    idxs = find_peaks(ref_yy)[0]
    idxs = idxs[np.argsort(x[idxs])]  # sort by increasing energy
    if n_peaks is not None:
        idxs = idxs[:n_peaks]

    crit_list = [[i, x[i], ref_yy[i]] for i in idxs]
    return np.array(crit_list)
    
def old_fwhm_finder(x, ref_yy, x_max):
    """
    Estimate the full width at half maximum (FWHM) around a given x_max.

    Parameters
    ----------
    x : array_like
        x-values of the signal.
    ref_yy : array_like
        y-values of the signal.
    x_max : float
        x-position near the peak of interest.

    Returns
    -------
    fwhm : float
        Full width at half maximum.
    i_l : int
        Index of the left half-max point.
    i_r : int
        Index of the right half-max point.
    """
    data = np.array([x, ref_yy], dtype='float').T
    data = data[np.argsort(data[:,0])]
    x = data.T[0]
    y = data.T[1]
    
    max_i = np.argmin(np.abs(x - x_max))
    
    hm = y[max_i]/2
    
    i_l = np.argmin(np.abs(y[:max_i]-hm))
    i_r = np.argmin(np.abs(y[max_i+1:]-hm)) + max_i + 1
    fwhm = x[i_r] - x[i_l]
    return [fwhm, i_l, i_r]


def func_compositor(operation, n_args, *funcs):
    '''
    n_args(list): len(n_args) must be equal to len(funcs); n_args[i] is the number of arguments required by the i-th function combined
    '''
    
    def composite_func(x, *args):
        results = []
        arg_cntr = 0
        for i in range(len(funcs)):
            curr_args = args[arg_cntr:arg_cntr+n_args[i]]
            results.append(funcs[i](x, *curr_args))
            arg_cntr += n_args[i]
        results = np.array(results)
        return operation(results)
    return composite_func


dict_fnames = {
    'gaussian_2D': gaussian_2D,
    'line_2D': line_2D,
    'lorentzian_2D': lorentzian_2D
}

dict_print_names = {
    'gaussian_2D': 'Gaussian',
    'lorentzian_2D': 'Lorentzian',
    'line_2D': 'Linear'
}

dict_nargs = {
    'gaussian_2D': 3,  # [pref, mean, std]
    'line_2D': 2,       # [a, b]
    'lorentzian_2D': 3 # [area, center, gamma]
}

dict_area = {
    'gaussian_2D': 0,
    'lorentzian_2D': 0
}

dict_centers = {
    'gaussian_2D': 1,
    'lorentzian_2D': 1
}

dict_width = {
    'gaussian_2D': 2,
    'lorentzian_2D': 2
    
}


def stable_cubic_spline(x_p, y_p, bc_type='not-a-knot', ltl=1, rtl=1, lts=1, rts=1):   
    """
    Create a stabilized cubic spline interpolator by adding flat tails to the input data.

    This function extends the input data with constant-value tails on both the left and
    right sides, which helps reduce extrapolation artifacts and ensures more stable
    behavior of the cubic spline near the boundaries.

    Parameters
    ----------
    x_p : array_like
        1D array of x-values for the original data points. Must be increasing.
    y_p : array_like
        1D array of y-values corresponding to `x_p`.
    bc_type : str or 2-tuple, optional
        Boundary condition passed to `scipy.interpolate.CubicSpline`. Default is 'not-a-knot'.
    ltl : float, optional
        Length of the left tail (in x-units). Default is 1.
    rtl : float, optional
        Length of the right tail (in x-units). Default is 1.
    lts : float, optional
        Step size used to discretize the left tail. Default is 1.
    rts : float, optional
        Step size used to discretize the right tail. Default is 1.

    Returns
    -------
    scipy.interpolate.CubicSpline
        A cubic spline object defined over the extended domain with stabilized tails.

    Notes
    -----
    The added tails have constant y-values equal to the first and last values of `y_p`.
    This is useful to avoid spline overshoot or unstable extrapolation near the edges.

    Examples
    --------
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 0])
    >>> spline = stable_cubic_spline(x, y, ltl=2, rtl=2)
    >>> spline(-1), spline(1), spline(3)  # evaluate in and outside the original range
    """
    
    # create the tails
    lt = np.arange(0, ltl, lts) - ltl + x_p[0] # left tail
    rt = np.arange(0, rtl, rts) + x_p[-1] + rts # right tail
    y_lt = np.ones(len(lt))*y_p[0]
    y_rt = np.ones(len(rt))*y_p[-1]

    # append the tails
    x = np.append(np.append(lt, x_p), rt)
    y = np.append(np.append(y_lt, y_p), y_rt)
    

    sp = CubicSpline(x, y, bc_type=bc_type)
    return sp



def deconvolve_spectrum(x,
                        y, 
                        nfuncs, 
                        func_names, 
                        check_npeaks=False,
                        n_peaks=None,
                        remove_baseline=True,
                        baseline_sampling=None,
                        offset=True,
                        mode='sequential',
                        centers=False,
                        width=False, 
                        prefactor=False, 
                        prec_par=3,
                        print_pars=True,
                        plot_things=True):
    '''Deconvolve spectrum with custom functions.

    Parameters
    ----------
    x: 
    '''

    if check_npeaks is True:
        crit_list = peak_finder(x, y)
        if len(crit_list) > nfuncs:
            print(f'More then {nfuncs} peaks have been found, that is {len(crit_list)}, so {len(crit_list)-nfuncs} extra {dict_print_names[func_names[-1]]} functions will be used.')
            func_names.extend(repeat(func_names[-1], len(crit_list)-nfuncs))
            nfuncs = len(crit_list)

    if remove_baseline is True:
        #BASELINE
        if baseline_sampling == None:
            raise ValueError('An array or list with the sampling x values for the baselines must be given!')
        elif not isinstance(baseline_sampling, (list, np.ndarray)):
            raise TypeError('baseline_sampling must be an array or list!')
        # create baseline
        i_p = find_indices(np.array(baseline_sampling), x)
        #i_p = np.array([1, 10, 15, 50, 110])

        x_p = np.zeros(len(i_p))
        for i in range(len(i_p)):
            x_p[i] += x[i_p[i]]

        y_p = np.zeros(len(i_p))
        for i in range(len(i_p)):
            y_p[i] += y[i_p[i]]

        sp = stable_cubic_spline(x_p, y_p)
        y_bl = sp(x)
        #plt.plot(x, y_bl) # plot the baseline

        # subtract baseline
        x_sp = x.copy()
        y_sp = y.copy() - y_bl
    else:
        x_sp = x.copy()
        y_sp = y.copy()


    if offset is True:
        #OFFSET
        offset = min(y_sp)
        y_sp -= offset



    #### FIT ####
    if mode == 'sequential':
        pars = [[random.random() for _ in range(dict_nargs[func_name])] for func_name in func_names] # generate all random pars
        residue = y_sp.copy()
        opt_pars = []
        for i in range(nfuncs):
            # find the highest peak in the residue
            crit_point = peak_finder(x_sp, residue)
            print(f'crit point for f.1: {crit_point}')
            print(f'crit_point[:,2]: {crit_point[:,2]}')
            crit_point = crit_point[np.argsort(crit_point[:,2])[::-1]][0] # [i, x[i], y[i]]
            #print(f'crit point for f.1: {crit_point}')
            if centers == True:
                # modify the "center" parameter of the function in order to make it align with the highest peak in the current residue
                # 
                # first, find the position of the center parameter in the list of paramters for the current function
                c_i = dict_centers[func_names[i]] 
                # second, convert the peak center (max position) into the the center parameter of the model function
                curr_center_par = crit_point[1] # to be adapted to functions
                # fourth, change the center parameter
                pars[i][c_i] = curr_center_par

            if width == True:
                # modify the "width" parameter of the function in order to make it equal to the FWHM of the peak
                #
                # first, find the position of the width parameter in the list of paramters for the current function
                w_i = dict_width[func_names[i]]  
                
                fwhm = peak_widths(y_sp, peaks=np.array([crit_point[0]], dtype='int'))[0][0] # 4, n_peaks (we want the first info, which is the width, and n_peaks=1)

                # third, change the center parameter
                pars[i][w_i] = fwhm

            if prefactor == True:
               # modify the "prefactor" (area) parameter of the function in order to make it equal to the height of the peak
                #
                # first, find the position of the width parameter in the list of paramters for the current function
                p_i = dict_area[func_names[i]]   
                pref = crit_point[2]
                # third, change the area parameter
                pars[i][p_i] = pref

            func_name = func_names[i]
            curr_f = dict_fnames[func_name]
            curr_ref = residue # curr_ref is the residue from the previous optimization

            # launch the optimization of the i-th curve
            n_trials_max = 5000
            n_trials = 0
            popt = pars[i].copy()
            while True:
                n_trials += 1
                try:
                    popt, pcov = curve_fit(curr_f, x_sp, curr_ref, p0=popt)
                    # compute the error
                    err = np.abs(curr_ref - curr_f(x_sp, *popt)).mean()
                    if err < 0.005:
                        #print('Found the best')
                        break
                except Exception as e:
                    print('Exception!')
                if n_trials >= n_trials_max:
                    print(f'Max number of trials reached for function n. {i+1}!')
                    break
            opt_pars.append(popt)
            residue = curr_ref - curr_f(x_sp, *popt)
            #plt.plot(x_sp, curr_f(x_sp, *popt))
            #plt.plot(x_sp, residue)

        final_res = residue
        final_fit = np.array([dict_fnames[func_name](x_sp, *parameters) for func_name, parameters in zip(func_names, opt_pars)]).sum(axis=0)

    elif mode == 'simultaneous':
        raise ValueError('The simultaneous mode is not available so far. Sorry for this :(')
        pars = []
        for i in range(nfuncs):
            pars.extend([random.random(), random.random(), random.random()])
        #pars = np.array(pars) + np.array([5.69312704, 382.71396647, 2.81116621,   5.32071454, 410.96052775, 1.43683827])
        #pars[1] = 382
        #pars[4] = 410
        #for i in range(len(crit_list)):
        #    pars[i*3 + 0] = ref_yy[crit_list[i]] # prefactor
        #    pars[i*3 + 1] = x[crit_list[i]] # mean


        # create the function
        old_nargs = 0
        for i, func_name in enumerate(func_names):
            new_func = dict_fnames[func_name]
            new_nargs = dict_nargs[func_name]
            if i == 0:
                curr_f = new_func
            else:
                curr_f = func_compositor(sum_operation, [new_nargs, old_nargs], new_func, curr_f)
            old_nargs += new_nargs


        n_trials_max = 1000
        n_trials = 0
        popt = pars.copy()
        while True:
            n_trials += 1
            try:
                popt, pcov = curve_fit(curr_f, x_sp, y_sp, p0=popt)

                # compute the error
                err = np.abs(y_sp - curr_f(x_sp, *popt)).mean()
                #print(err)
                if err < 0.005:
                    print('Found the best')
                    break
            except Exception as e:
                pass
            if n_trials >= n_trials_max:
                print('Max number of trials reached!')
                break
        final_fit = curr_f(x_sp, *popt)


    if print_pars is True:
        #### WRITE THE PARAMETERS ####
        txt = f'##############\n'
        txt += f'   RESULTS\n'
        txt += f'##############\n\n'

        for i in range(nfuncs):
            txt += f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n'
            txt += f'Function {i+1}: {dict_print_names[func_names[i]]}\n'
            for j in range(len(opt_pars[i])):
                txt += f'\t-par. {j} = {opt_pars[i][j]:0.{prec_par}f}\n'
            txt += f'------------------------------------------------------------\n'
            txt += f'\n'

        print(txt)

    if plot_things is True:
        #### PLOT THE REFERENCE DATA ####
        plt.figure()
        plt.plot(x_sp, y_sp, label='Ref. data')
        plt.plot(x_sp, final_fit, label='Fit')
        plt.plot(x_sp, final_res, label='Residue')

        if remove_baseline == True:
            plt.figure()
            plt.plot(x_sp, y_sp+y_bl, label='Ref. data')
            plt.plot(x_sp, y_bl-offset, label='baseline')
            plt.plot(x_sp, final_fit+y_bl, label='Fit')
            plt.plot(x_sp, final_res, label='Residue')

        plt.legend()

        plt.figure()
        for i in range(nfuncs):
            plt.plot(x_sp, dict_fnames[func_names[i]](x_sp, *opt_pars[i]), label=f'Function n. {i+1}')
        plt.legend()
    return opt_pars

