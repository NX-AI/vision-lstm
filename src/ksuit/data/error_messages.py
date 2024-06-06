def getshape_instead_of_getdim(getdim_names):
    getshape_names = [f"getshape_{getdim_name[len('getdim_'):]}" for getdim_name in getdim_names]
    return f"implement 'getshape' instead of 'getdim' (expected {getshape_names} but found {getdim_names})"
