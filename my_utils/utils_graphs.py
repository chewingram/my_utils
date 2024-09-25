import numpy as np
import os
import sys
from ase.io import read, write
import matplotlib.pyplot as plt
import os
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from utils import cap_first, mae, rmse, R2, low_first, path



def plot_correlation_graph(data1,
                           data2,
                           prop_name='property',
                           ref_prop_label='Reference',
                           pred_prop_label='Predicted',
                           plot_title=None,
                           units=None,
                           make_file=False,
                           file_path='comparison.dat',
                           offset=None,
                           save=True,
                           save_dir='./',
                           save_name='figure',
                           save_formats=['png'],
                           save_kwargs=[dict(bbox_inches='tight', dpi=600)]):
    '''
    Function to plot the correlation graph for Energy, Forces and Stress.
    Args:
    data1(list, numpy.array): list or numpy array containing the reference (true) data for the property of interest;
    data2(list, numpy.array): list or numpy array containing the predicted data for the property of interest;
    prop_name(str): name of the property;
    ref_prop_label(str): label to put before the name of the reference property in the axis (x); e.g. 'DFT' or 'True';
    pred_prop_label(str): label to put before the name of the predicted property in the axis (y); e.g. 'Predicted';
    plot_title(str or None): title for the plot (optional);
    units(str): units to write the property in;
    make_file(bool): True: create a comparison file;
    file_path: path to the comparison file to save;
    save(bool): True: save the plot;
    save_name(str): name of the file to save; NO extension: it will be added later (see save_formats);
    save_formats(list): list of formats of the images to save; they must be compatible with matplotlib (see matplotlib docs);
    save_kwargs(list): list of dicationary for each format given in "formats" containing the additional kwargs that will be passed to
                       pyplot.savefig() (see matplotlib docs).
    '''


    if isinstance(data1, list):
        data1 = np.array(data1)
    elif isinstance(data1, type(np.array([1]))):
        pass
    else:
        raise TypeError('The reference data must be given as a list or numpy array!')

    if isinstance(data2, list):
        data2 = np.array(data2)
    elif isinstance(data2, type(np.array([1]))):
        pass
    else:
        raise TypeError('The predicted data must be given as a list or numpy array!')

    if len(data2) != len(data1):
        raise ValueError('The reference and the predicted data must be the same lenght!')

    if units == None:
        units = 'no units'

    if not isinstance(save_formats, list):
        raise TypeError('The saving formats (save_formats) must be passed as a list of strigs!')

    if not isinstance(save_kwargs, list):
        raise TypeError('The additional kwargs for each format (save_kwargs) must be passed as a list of dictionaries!')

    if len(save_formats) != len(save_kwargs):
        raise ValueError('The number of formats must coincide with the number of additional kwargs to save the image!')

    if save == True:
        save_dir = path(save_dir)

    # Compute errors and write data on files
    mae2 = mae(data1, data2)
    rmse2 = rmse(data1, data2)
    R22 = R2(data1, data2)
    errs = [rmse2, mae2, R22]

    if make_file == True:
        text = f'# rmse: {rmse2:.5f} {units},    mae: {mae2:.5f} {units}    R2: {R22:.5f}\n'
        text += f'#  True {low_first(prop_name)}           Predicted {low_first(prop_name)}\n'
        for x, y in zip(data1, data2):
            text += f'{x:.20f}  {y:.20f}\n'
        with open(file_path, 'w') as fl:
            fl.write(text)

    if offset is None:
        offset = 0

    txt = f"rmse: {errs[0]:.4f} {units}\nmae: {errs[1]:.4f} {units}\nR$^2$: {errs[2]:.5f}"
    x_data = data1
    y_data = data2

    n = 2.3
    fig1 = plt.figure(figsize=(n, n))
    ax = fig1.add_axes((0, 0, 1, 1))
    ax.plot(x_data, y_data, ".", markersize=10, mew=0.6, mec="#00316e", mfc='#70c8ff')

    # plot bisector line
    coords = (min(np.concatenate((x_data, y_data))), max(np.concatenate((x_data, y_data))))
    ax.plot(coords, coords)

    # add labels and title
    ax.set_xlabel(f"{ref_prop_label} {prop_name} ({units})")
    ax.set_ylabel(f"{pred_prop_label} {prop_name} ({units})")
    if plot_title != None:
        ax.set_title(plot_title)

    # add textbox with errors
    tbox = ax.text(min(np.concatenate((x_data, y_data))),
                   max(np.concatenate((x_data, y_data))) - offset,
                   txt,
                   fontweight='bold',
                   horizontalalignment='left',
                   verticalalignment='top',)
    #bbox = tbox.get_window_extent() # get the bounding box of the text
    #transform = ax.transData.inverted() # prepare the transformation into data units
    #bbox = transform.transform_bbox(bbox) # transform it into data units

    # save
    if save == True:
        for i, form in enumerate(save_formats):
            plt.savefig(fname=f'{save_dir}{save_name}.{form}', format=form, **(save_kwargs[i]))

