"""
Utility functions.
"""

def update_sptype(sptypes):
    """
    Function to update a list with spectral types to two characters (e.g., M8, L3, or T1).

    :param sptypes: Spectral types.
    :type sptypes: numpy.ndarray

    :return: Updated spectral types.
    :rtype: numpy.ndarray
    """

    mlty = ('M', 'L', 'T', 'Y')

    for i, spt in enumerate(sptypes):
        if spt == 'None':
            pass

        elif spt == 'null':
            sptypes[i] = 'None'

        else:
            for item in mlty:
                try:
                    sp_index = spt.index(item)
                    sptypes[i] = spt[sp_index:sp_index+2]

                except ValueError:
                    pass

    return sptypes
