# Put nwb raw data file here
- "/acquisition/timeseries/broadband/data" - k x n
    - The broadband neural recordings.
- "/acquisition/timeseries/broadband/data/conversion" (scalar attribute)
    - When multiplied by each sample converts the data into units of volts.
- "/acquisition/timeseries/broadband/timestamps" - k x 1
    - Timestamps for each sample, seconds.
- "/general/extracellular_ephys/electrode_map" - n x 3
    - The relative coordinates of each electrode contact (x, y, z), meters.