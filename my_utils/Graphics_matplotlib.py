import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/longosamuel/Work/my_scripts')
from utils import warn, repeat
from numbers import Number

class Histogram:
    '''
    Class for histograms.
    Parameters:
    y_data (int): a list or numpy.array containing the numeric values to group into bins 
    '''
    
    def __init__(self, y_data):
        msg = "The data to be histogram-fied should be given as a list (or numpy array) of numbers."
        if not isinstance(y_data, list) and not isinstance(y_data, type(np.array([]))):
            raise TypeError(msg)
        elif not all([isinstance(x, Number) for x in y_data]):
            raise TypeError(msg)
        self.y_data = y_data
        self.figs = []
        self.axes = []
        self.patches = []
        
    def histofy(self, mode='n_bins', nbins=None, bins=None, normalized=False):
        '''
        Function to create an histogram. 
        There are three modes:
            - n_bins: the number of bins is given and the bins will be created automatically. The end of each bin is halfway    
                      between its center and the next bin's one. The first bin left boundary is the lowest value in y_data if 
                      it's less than the center of the bin, otherwise it is at the same distance from the center as the right
                      boundary. Analogously for the last bin. 
            - custom_bins: the bins are given directly. They must be a list or numpy.array of list (or numpy.array) containing 
                           the left and right boundary of the bin. Ex.: [[1,4], [4,7], [9,12]].
            - custom_centers: the centers of the bins are given as a list or numpy.array. The bins will be reconstructed 
        Parameters:
        mode (str): 
        nbins (int):
        bins (list or np.array):
        normalized (bool): 
        '''
        y_data = self.y_data
        self.normalized = normalized
        self.tot_elems = len(y_data)
        if normalized == True:
            norm = self.tot_elems
        else:
            norm = 1
 
        if mode != 'n_bins' and type(nbins) != type(None):
            warn(f"The mode is {mode}, so the number of bins given as parameter will be ignored.")
        if 'custom' not in mode and type(bins) != type(None):
            warn(f"The mode is {mode}, so the bins given as parameter will be ignored")

        if mode == 'n_bins':
            if type(nbins) != int:
                raise TypeError("For the 'nbins' mode an integer must be given as number of bins.")
            h = (max(y_data)-min(y_data))/nbins
            bins = repeat([None,None], nbins)
            bin_centers = [None]
            #first bin center
            bin_centers[0] = min(y_data) + h / 2
            #bin centers in the middle and last one
            bin_centers.extend([bin_centers[0] + 2 * i * (h / 2) for i in range(1, nbins)])

            #bins
            bins = [[x-(h/2), x+(h/2)] for x in bin_centers]
#             bins[0] = [min(y_data), min(y_data) + h]
#             if min(y_data) <= 2 * bin_centers[0] - bins[0][1]:
#                 bins[0][0] = min(y_data)
#             else:
#                 bins[0][0] = 2 * bin_centers[0] - bins[0][1]

#             #bins in the middle and last one
#             for i in range(1, nbins):
#                 bins[i] = [bins[i-1][1], bins[i-1][1] + h]

        elif mode == 'custom_bins':
            msg="In the custom_bins mode the bins must be specified as a list (or np array) of lists (or np arrays). " +\
            "Each internal list must contain the two extremes of the bin."
            if not (isinstance(bins, list) or isinstance(bins, type(np.array([])))):
                raise TypeError(msg)
            elif not all([isinstance(x, list) or isinstance(x, type(np.array([]))) for x in bins]):
                raise TypeError(msg)
            elif not all([isinstance(y, Number) for x in bins for y in x]):
                raise TypeError(msg)
            if isinstance(bins, type(np.array([]))):
                bins = bins.tolist()
            # sort the pairs of extremes (on the extremes level)
            for x in bins:
                x.sort()
            # sort the bins (on the low-extreme level)
            bins.sort(key=lambda p : p[0])

            # check that there are not overlaps between bins 
            for i in range(len(bins)):
                for j in range(i+1, len(bins)):
                    if bins[i][1] <= bins[j][0] or bins[j][1] <= bins[i][0]:
                        pass
                    else:
                        raise ValueError("Two or more bins provided overlap! Please give some non-overlapping bins.")

        elif mode == 'custom_centers':
            msg = "In the custom_centers mode the centers of the bins must be specified as a list (or np array) of " +\
                  " float or int."
            if not (isinstance(bins, list) or isinstance(bins, type(np.array([])))):
                raise TypeError(msg)
            elif not all([isinstance(x, Number) for x in bins]):
                raise TypeError(msg)
            if isinstance(bins, type(np.array([]))):
                bin_centers = bins.tolist()
            bin_centers = bins
            bin_centers.sort()
            nbins = len(bin_centers)
            bins = repeat([None, None], nbins)
            #first bin 
            bins[0][1] = bin_centers[0] + (bin_centers[1] - bin_centers[0]) / 2
            if min(y_data) <= 2 * bin_centers[0] - bins[0][1]:
                bins[0][0] = min(y_data)
            else:
                bins[0][0] = 2 * bin_centers[0] - bins[0][1]
            #bins in the middle
            for i, x in enumerate(bin_centers[1:-1]):
                bins[i+1][0] = bins[i][1]
                bins[i+1][1] = x + (bin_centers[i+2] - x) /2    
            #last bin
            bins[-1][0] = bins[-2][1]
            if max(y_data) >= 2 * bin_centers[-1] - bins[-1][0]:
                bins[-1][1] = max(y_data)
            else:
                bins[-1][1] = 2 * bin_centers[-1] - bins[-1][0]    

        if mode != 'custom_centers':
            bin_centers = np.array([x[0] + (x[1]-x[0])/2 for x in bins])

        hist = [[x, y, 0] for x, y in zip(bins, bin_centers)]
        
        for i, x in enumerate(bins):
            hist[i][2] = len([y for y in y_data if y >= x[0] and y < x[1]])/norm

        # check if there are values == to the right extreme of the higher bin
        for y in y_data:    
            if y == bins[-1][1]:
                    hist[-1][2] = hist[-1][2] + 1/norm

        self.hist = hist
    
    def make_bars(self, plot=True, save_plot=False, *args, **kwargs):
        plot_elements = []
        bins = [x[0] for x in self.hist]
        patches = []
        for i, x in enumerate(self.hist):
            vertices = np.array([(x[0][0], 0), (x[0][0], x[2]), (x[0][1], x[2]), (x[0][1], 0)], object)# clockwise from the bottom-left
            patches.append(matplotlib.patches.Polygon(vertices, edgecolor='black'))
        plot_elements.extend(patches)
        
        self.plot_elements = plot_elements  
        self.patches = patches
        
        if plot == True:
            fig1, ax1 = self.plot_bars(save=True, **kwargs)
            return fig1, ax1
        
        
    def plot_bars(self, save=False, **kwargs):
        '''
        Function to plot the bars.
        Args:
        save(bool): False: the plot will not be saved; True: the plot will be saved (in this case a dictionary with the 
                           parametes required by matplotlib.pyplot.savefig() must be passed as "save_args"
        **kwargs(-): Arguments for the plot. Currently they are:
                     title: title of the plot; default= 'no title'
                     xlabel: label of the x-axis; default= 'no label'
                     ylabel: label of the y-axis; default= 'no label'
                     save_args: dictionary of parameters to pass to matplotlib.pyploy.savefig(); default= dict()            
        '''
        args_plot = dict(title = 'no title',
                         xlabel = 'no label',
                         ylabel = 'no label',
                         save_args = dict())
        
        for key in args_plot.keys():
            if key in kwargs.keys():
                args_plot[key] = kwargs[key]
            
        
#         if 'title' in kwargs.keys():
#             title = kwargs['title']
#         else:
#             title = 'no title'
    
#         if 'xlabel' in kwargs.keys():
#             xlabel = kwargs['xlabel']
#         else:
#             xlabel = 'no label'
    
#         if 'ylabel' in kwargs.keys():
#             ylabel = kwargs['ylabel']
#         else:
#             ylabel = 'no label'
#         if 'save_args' in kwargs.keys():
#             save_args = kwargs['save_args']
#         else:
#             save_args = dict()
        
        fig1 = plt.figure()
        ax = fig1.add_axes((0,0,1,1))
        for x in self.patches:
            ax.add_patch(x)
        ax.set_xlim(self.hist[0][0][0], self.hist[-1][0][1])
        ax.set_ylim(0, max([x[2] for x in self.hist]))
        ax.set_xticks([x[1] for x in self.hist[::15]])
        ax.set_title(plot_args['title'])
        ax.set_xlabel(plot_args['xlabel'])
        ax.set_ylabel(plot_args['ylabel'])
        if save == True:
            plt.savefig(**plot_args['save_args'])
        return fig1, ax
