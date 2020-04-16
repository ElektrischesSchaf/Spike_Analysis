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
    ALL_List_FILE = os.listdir(FILE_PATH)
    ALL_List_FILE.sort()
    # List_FILE=ALL_List_FILE[:] # 3=indy_20160630_01, 7=indy_20160927_04, 16=indy_20161014_04, 19=indy_20161025_04, 29=indy_20170131_02
    bad_session_indices=[3, 7, 16, 19, 29]
    List_FILE=[]
    for i in range(30):
        if i not in bad_session_indices:
            List_FILE.append(ALL_List_FILE[i] )

    GET_FILE = []
    for FILE_NAME in List_FILE:
        GET_FILE.append(FILE_PATH + FILE_NAME)
