import numpy as np

def build_dct_unitary_dictionary(patch_size):
    
    # BUILD_DCT_UNITARY_DICTIONARY Creates an overcomplete 2D-DCT dictionary.
    #
    # Inputs:
    # patch_size  - Atom size [height, width] (must satisfy height == width)
    #
    # Outputs:
    # DCT - Unitary DCT dictionary with normalized columns
    
    # Make sure that the patch is square
    try:
        patch_size[0] != patch_size[1]
    except:
        print('This only works for square patches')

    nAtoms = patch_size[0]*patch_size[1]

    # Create DCT for one axis
    Pn = int(np.ceil(np.sqrt(nAtoms)))
    DCT = np.zeros((patch_size[0] , Pn))

    for k in range(Pn):
    
        V = np.cos( (0.5 + np.arange(Pn) ) * k * np.pi / Pn)
        if k>0:
            V = V - np.mean(V)
        DCT[: , k] = V / np.sqrt(np.sum(V**2))

    # Create the DCT for both axes
    DCT = np.kron(DCT , DCT)
    
    return DCT


