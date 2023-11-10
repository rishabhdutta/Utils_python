import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.path import Path

def remove_polygon_from_array(array, xmin = np.nan, xmax= np.nan, ymin= np.nan, ymax= np.nan):
    plt.imshow(array, cmap='bwr', origin='lower',interpolation='nearest')
        
    if not np.isnan(xmin):
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

    plt.colorbar()
    plt.title('Original Array')

    # Allowing user to draw a polygon on the plot
    polygon_path = plt.ginput(n=-1, show_clicks=True)
    plt.close()

    # Creating a Path object from the polygon points
    polygon_vertices = [point for point in polygon_path]
    polygon_path = Path(polygon_vertices)

    # Creating a mask to identify points within the polygon
    x, y = np.indices(array.shape)
    points = np.column_stack((y.flatten(), x.flatten()))
    mask = polygon_path.contains_points(points)
    mask = mask.reshape(array.shape)

    # Removing the area within the polygon by setting values to zero
    array[mask] = np.nan

    # Displaying the modified array
    plt.imshow(array, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('Modified Array')

    plt.show()