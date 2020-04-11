# Import math & graph plotting library
import math
import matplotlib.pyplot as plt

class TriangularWavePlotter(object):
    ''' This Class generates the data for getting Triangular Learning Rate as described in 
    paper https://arxiv.org/pdf/1506.01186.pdf

    Example Usage:
        a = TriangularWavePlotter(1000, 0.001, 0.08)
        a.generate_datapoints(num_curves = 3, draw_graph = True)
    '''

    def __init__(self, step_size, min_bound, max_bound):
        ''' Initialize the object with following parameters, 

        Args:
        self          : Object
        step_size     : Number of iterations for which 'learning rate' should increase. According to the 
                        paper, this is 2 to 10 times the number of iteration in an epoch
        min_bound     : The minimum learning rate is when model accuracy starts increasing in “LR range test” 
        max_bound     : The maximum learning rate is when model accuracy starts decreasing or plateus in 
                        “LR range test”
                 ^                            ___________ Upper Bound
              L  |          /\        /\                
              R  |         /  \      /  \
                 |        /    \    /    \
                 |    \	 /      \  /      \  /
                 |     \/        \/        \/ ___________ Lower Bound
                 |      | 		  |	   |
                 |      |< Cycle> |<SS>|
                 |			                * SS = Step Size
                -|-------------------------------------------------> Iterations

        '''
        self.step_size  = step_size
        self.min_bound  = min_bound
        self.max_bound  = max_bound

    def generate_datapoints (self, num_curves, draw_graph = False):
        '''This function generates the datapoints for the plotting the learning rate.

        Args:
        num_curves      : How many curves are required to be plotted?
        draw_graph      : If True, the graph will be plotted, else not  

        Returns:
        None
        '''
        self._iterations = self.step_size * 2 * num_curves
        self.plot_data = []

        # Iterate over the 'interations' and get the corresponding learning rate
        # Refer Paper for discription of following terms 
        for i in range(self._iterations):
            local_cycle = self.get_cycle(i, self.step_size)
            local_x     = self.get_x(i, local_cycle, self.step_size)
            local_LR    = self.get_LR(local_x, self.min_bound, self.max_bound)
            self.plot_data.append(local_LR)

        # If plotting is required
        if draw_graph:
            self.graph_plotter()
    
    def graph_plotter(self):
        ''' This function plots the graph of data-points stored in the object

        Args:
            Self
        
        Returns:
            None
        
        Displays:
            Plot
        '''
        plt.plot(self.plot_data)
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.grid('both')
        plt.title('Learing Rate Curve')
        plt.show()


    @staticmethod
    def get_cycle(i, step_size):
        '''This function returns the current cycle
        
        Args:
        i           : Current iteration
        step_size   : Refer __init__

        Returns:
        cycle       : The current cycle number
        '''
        cycle = math.floor(1+(i/(2*step_size)))
        return (cycle)

    @staticmethod
    def get_x(i, cycle_val, step_size):
        ''' This function gets the current x value. This will help in 
        understanding whether the live is assending or descending

        Args:
        i           : Current iteration
        cycle_val   : Current cycle
        step_size   : Refer class __init__

        Returns:
        x           : Current point on the 2D plane

        '''
        x = abs((i/step_size) - (2*cycle_val) + 1)
        return(x)

    @staticmethod
    def get_LR(local_x, min_bound, max_bound):
        ''' Calculates the LR based on the current 'x' on the 2D plane

        Args:
        min_bound, max_bound : Refer class __init__
        local_x              : Current datapoint on the 2D plane

        Returns:
        LR                   : Learning rate corresponding to the datapoint
        '''
        LR = (min_bound) + ((max_bound - min_bound)*(1-local_x))
        return (LR)