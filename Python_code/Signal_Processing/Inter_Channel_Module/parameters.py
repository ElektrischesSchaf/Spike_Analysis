import os
class my_parameters():
    channel_number=31
    start_second=300
    plot_time_duration=5
    end_second=start_second+plot_time_duration
    band_start=0.5
    band_cutoff=40
    session_name='indy_20161007_02'
    kinematic_variable_type='vel'

    # cross sessions
    FILE_PATH = '../../../Dataset/The_nwb_Raw_Dataset/'
    List_FILE = os.listdir(FILE_PATH)
    List_FILE.sort()
    List_FILE=List_FILE[:] # 3
    GET_FILE = []
    for FILE_NAME in List_FILE:
        GET_FILE.append(FILE_PATH + FILE_NAME)
