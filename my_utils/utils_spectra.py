import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact
import random
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import os
from pathlib import Path

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



def sum_operation(args):
    return sum(args)
    
def boh(args):
    return args[0] + 3* (args[1])

def peak_finder(x, ref_yy):
    crit_list = []
    for i in range(1, len(ref_yy)-1):
        if ref_yy[i]>= ref_yy[i-1] and ref_yy[i] >= ref_yy[i+1]:# or ref_yy[i]<= ref_yy[i-1] and ref_yy[i] <= ref_yy[i+1]:
            crit_list.append([i, x[i], ref_yy[i]])
    return np.array(crit_list)
    
def fwhm_finder(x, ref_yy, x_max):
    '''
    '''
    data = np.array([[z,y] for z, y in zip(x, ref_yy)], dtype='float')
    data = data[np.argsort(data[:,0])]
    x = data.T[0]
    y = data.T[1]
    
    max_i = np.argmin(np.array([abs(x-x_max)]))
    x_max = x[max_i]
    y_max = y[max_i]
    hm = y_max/2
    
    i_l = np.argmin(np.array([abs(y[:max_i]-hm)]))
    i_r = np.argmin(np.array([abs(y[max_i+1:]-hm)])) + max_i + 1
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
                        remove_baseline=True,
                        offset=True,
                        mode='sequential',
                        centers=False,
                        width=False, 
                        prefactor=False, 
                        prec_par=3,
                        print_pars=True,
                        plot_things=True):

    if check_npeaks is True:
        crit_list = peak_finder(x_sp, y_sp)
        if len(crit_list) > nlor:
            print(f'More then {nlor} peaks have been found, that is {len(crit_list)}, so {len(crit_list)} lorentzians will be used.')
            nlor = len(crit_list)

    if remove_baseline is True:
        #BASELINE
        # create baseline
        i_p = np.array([1, 10, 15, 50, 110])

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


    if offset is True:
        #OFFSET
        y_sp -= min(y_sp)



    #### FIT ####
    if mode == 'sequential':
        pars = [[random.random() for _ in range(dict_nargs[func_name])] for func_name in func_names] # generate all random pars
        residue = y_sp.copy()
        opt_pars = []
        for i in range(nfuncs):

            if centers == True:
                # modify the "center" parameter of the function in order to make it align with the highest peak in the current residue
                # 
                # first, find the position of the center parameter in the list of paramters for the current function
                c_i = dict_centers[func_names[i]] 
                # second, find the peaks and take point with the maximum value of y 
                crit_list = peak_finder(x_sp, residue)
                curr_center = crit_list[np.argmax(crit_list[:,2])]
                # third, change the center parameter
                pars[i][c_i] = curr_center[1]

            if width == True:
                # modify the "width" parameter of the function in order to make it equal to the FWHM of the peak
                #
                # first, find the position of the width parameter in the list of paramters for the current function
                w_i = dict_width[func_names[i]]  
                # second, find the peaks and take point with the maximum value of y 
                crit_list = peak_finder(x_sp, residue)
                curr_center = crit_list[np.argmax(crit_list[:,2])] # [index, x, y]
                fwhm = fwhm_finder(x_sp, residue, curr_center[1])[0] 
                # third, change the center parameter
                pars[i][w_i] = fwhm

            if prefactor == True:
               # modify the "prefactor" (area) parameter of the function in order to make it equal to the height of the peak
                #
                # first, find the position of the width parameter in the list of paramters for the current function
                p_i = dict_area[func_names[i]]   
                # second, find the peaks and take point with the maximum value of y 
                crit_list = peak_finder(x_sp, residue)
                curr_center = crit_list[np.argmax(crit_list[:,2])] # [index, x, y]
                pref = curr_center[2]
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
            txt += f'Function 1: {dict_print_names[func_names[i]]}\n'
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

        plt.legend()

        plt.figure()
        for i in range(nfuncs):
            plt.plot(x_sp, dict_fnames[func_names[i]](x_sp, *opt_pars[i]), label=f'Function n. {i+1}')
        plt.legend()
    
    return opt_pars


