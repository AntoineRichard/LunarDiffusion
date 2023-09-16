# Lunar Zoomer DEM Diffuser

# Misc Notes:

- H5Py memory leak when converting to pytorch tensor.
  See: https://github.com/h5py/h5py/issues/2010
  Install with conda which comes with its own hdf5 library.
  Or install compile against local libhdf5 using `pip install --no-binary=h5py h5py`
