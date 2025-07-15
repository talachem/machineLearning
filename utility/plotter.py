#import matplotlib.pyplot as plt
import os


class Plotter():
    """
    This class can save matplotlib plots to a specified directory.
    """
    def __init__(self, directory='plots'):
        """
        Initialize the PlotWriter class.
        """
        self.directory = directory

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __call__(self, fig, filename, display=False):
        """
        Save the given matplotlib figure to a file and optionally display it.
        """
        try:
            filepath = os.path.join(self.directory, filename)
            fig.savefig(filepath)

            if display:
                pass
                #plt.show()

            #plt.close(fig)  # Close the figure
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")
