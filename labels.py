import numpy as np
import functional_functions as ff

""""
If you have an index k in the vector and want to know its position i,j in the square matrix, 
i, j = rows[k], cols[k]

Then, use those indices (i, j) to look up the corresponding labels  

labels[i], labels[j]
"""

def get_labels_of_index(k):
    """
    from index k in the vectorized correlation matrix, find the row&col labels from the original square matrix.
    NOTE: THE DIAGONAL IS NOT INCLUDED
    taks an integer index 
    returns a string 
    """
    # create vector of 360 labels 
    lut = ff.create_mmp_lookup()
    labels = np.array(lut["name"])[1:361]

    rows, cols = np.tril_indices(len(labels), k=-1)

    # find original indices i, j in the square matrix 
    i, j = rows[k], cols[k]

    # find original labels
    roi1 = labels[i]
    roi2 = labels[j]

    #return the results as a string
    return f"ROI 1: {roi1} | ROI 2: {roi2}"