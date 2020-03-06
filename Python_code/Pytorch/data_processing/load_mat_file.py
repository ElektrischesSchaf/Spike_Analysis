import numpy as np
import time
import h5py
from sklearn import datasets, svm, metrics
from sklearn.metrics import mean_squared_error, r2_score

class mat_file_processing():

    def histc(self, X, bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return r

    def get_spike_bins_matrix(self, the_file_name, the_sampling_rate, include_hash_unit):
        with h5py.File(the_file_name, 'r') as mat_file:        
            time_stamp=mat_file['t']  
            # or
            # time_stamp=mat_file.get('t')
            # time_stamp=np.array(time_stamp)
            # time_stamp.shape = (1, 204446)
            spikes = mat_file['spikes']
            firing_rate_cell=[[]]
            

            print('spikes shape: ', spikes.shape) #  (3, 192) in indy_20160407_02
            channel_number=int(spikes.shape[1] / 2) # 96 in indy_20160407_02
            numpy_finger_pos_1=np.empty([])
            numpy_finger_pos_GET_1=mat_file.get('finger_pos')
            numpy_finger_pos_1=np.array(numpy_finger_pos_GET_1)
            print('numpy_finger_pos_1.shape: ', numpy_finger_pos_1.shape) # (3, 204446) in indy_20160407_02
            
            finger_z_coor=numpy_finger_pos_1[0][:]
            finger_x_coor=numpy_finger_pos_1[1][:]
            finger_y_coor=numpy_finger_pos_1[2][:]

            x_position_label=[]
            y_position_label=[]
            z_position_label=[]

            x_velocity_label=[]
            y_velocity_label=[]
            z_velocity_label=[]

            x_acceleration_label=[]
            y_acceleration_label=[]
            z_acceleration_label=[]

            time_stamp_64ms=[]

            sampling_rate=the_sampling_rate # because 64ms

            #duration=1000
            duration=time_stamp.shape[1]
            sampling_index=0

            # Too slow app. 70 seconds
            ''' 
            while sampling_index < duration:
                #print('sampling_index = ', sampling_index)
                print( 'Progress of making sampling array: '+ str(   round( (sampling_index / duration)*100, 3)   )+' %' )
                time_stamp_64ms.append(time_stamp[0][sampling_index])
                sampling_index+=sampling_rate
            '''
            time_stamp_64ms=time_stamp[0][::sampling_rate]  # way faster, app. 4 seconds
            print('lenght of time_stamp_64ms: ', len(time_stamp_64ms)) # 12778 in indy_20160407_02
            
            testing_data_index=int(int(len(time_stamp_64ms))*0.8) # split 80% into training
            print('testing_data_index= ',testing_data_index) # 10222 in indy_20160407_02.mat

            # make x, y, z position label matrix with the sampling_rate
            '''
            index_label=0
            while index_label < duration:
                x_position_label.append(finger_x_coor[index_label] )
                y_position_label.append(finger_y_coor[index_label] )
                z_position_label.append(finger_z_coor[index_label] )
                index_label+=sampling_rate
            '''
            x_position_label=finger_x_coor[::sampling_rate] 
            x_position_label=x_position_label[1:] # [:-1] original, match Chieh's result
            y_position_label=finger_y_coor[::sampling_rate]
            y_position_label=y_position_label[1:]
            z_position_label=finger_z_coor[::sampling_rate]
            z_position_label=z_position_label[1:]
            print('Position label arrays finished')

            # Making spike counts matrix
            for channel_index in range(channel_number):
                #print('Channel progress: ' + str( round( (channel_index/channel_number)*100, 3) )+' %' ) # 96 channels in this dataset
                
                temp_spike_cell_1=[]
                temp_spike_cell_2=[]
                temp_spike_cell_3=[]

                temp_spike_cell_1=mat_file[ ( spikes[0][channel_index] ) ][()]
                temp_spike_cell_2=mat_file[ ( spikes[1][channel_index] ) ][()]
                temp_spike_cell_3=mat_file[ ( spikes[2][channel_index] ) ][()]

                temp_spike_cell_1=np.asarray(temp_spike_cell_1)
                temp_spike_cell_2=np.asarray(temp_spike_cell_2)
                temp_spike_cell_3=np.asarray(temp_spike_cell_3)

                time_stamp_64ms=np.asarray(time_stamp_64ms)

            
                temp_spike_cell_1=temp_spike_cell_1.flatten()
                temp_spike_cell_2=temp_spike_cell_2.flatten()
                temp_spike_cell_3=temp_spike_cell_3.flatten()
                time_stamp_64ms=time_stamp_64ms.flatten()
                
                '''
                print('shape of temp_spike_cell_1: ',temp_spike_cell_1.shape) # (5595,) in channel 1, indy_20160407_02
                print('shape of temp_spike_cell_2: ',temp_spike_cell_2.shape) # (2,) in channel 1, indy_20160407_02
                print('shape of temp_spike_cell_3: ',temp_spike_cell_3.shape) # (2,) in channel 1, indy_20160407_02
                print('shape of time_stamp_64ms: ',time_stamp_64ms.shape)
                '''

                if temp_spike_cell_1.shape[0] != 2 and include_hash_unit==True:

                    yee=self.histc(temp_spike_cell_1, time_stamp_64ms)
                    #print('shape of yee:  ',yee.shape)
                    firing_rate_cell.append(yee[:-1])
                    #print('yee: ',yee)

                else:
                    r = np.zeros( len(time_stamp_64ms)-1 )
                    firing_rate_cell.append(r)
                '''
                print('length of firing_rate in cell 1: ',end='')
                print(len(firing_rate_cell[:-1]))
                '''

                firing_rate_cell.append([])  

                if temp_spike_cell_2.shape[0] != 2:

                    # firing rate
                    yee=self.histc(temp_spike_cell_2, time_stamp_64ms)
                    #print('shape of yee:  ',yee.shape)
                    firing_rate_cell.append(yee[:-1])

                else:
                    r = np.zeros( len(time_stamp_64ms)-1 )
                    firing_rate_cell.append(r)

                '''
                print('length of firing_rate in cell 2: ',end='')
                print(len(firing_rate_cell[-1]))
                '''
                firing_rate_cell.append([])

                if temp_spike_cell_3.shape[0] != 2:
                    
                    # firing rate
                    yee=self.histc(temp_spike_cell_3, time_stamp_64ms)
                    #print('shape of yee:  ',yee.shape)
                    firing_rate_cell.append(yee[:-1])

                else:
                    r = np.zeros( len(time_stamp_64ms)-1 )
                    firing_rate_cell.append(r)
                '''
                print('length of firing_rate in cell 3: ',end='')
                print(len(firing_rate_cell[-1]))
                print('\n\n')
                '''
                firing_rate_cell.append([])

                '''
                print('row numbers of firing_rate_cell: ',end='')
                print( len( firing_rate_cell) )
                print('\n')
                '''
                '''
                for row_index in range( len( firing_rate_cell) ):            
                    print('length of firing_rate_cell['+ str(row_index) +']: ',end='')
                    print(len(firing_rate_cell[row_index]))
                print('\n')
                print('End of one channel '+ str(channel_index+1) +'\n') 
                '''
        return [firing_rate_cell, channel_number, testing_data_index, x_position_label, y_position_label, z_position_label]

    def get_labels(self, the_file_name, the_sampling_rate):
        with h5py.File(the_file_name, 'r') as mat_file:
            finger_pos = mat_file['finger_pos']
            time_stamp=mat_file['t']

            numpy_finger_pos_GET_2=mat_file.get('finger_pos')
            numpy_finger_pos_2=np.array(numpy_finger_pos_GET_2)

            numpy_time_stamp=mat_file.get('t')
            numpy_time_stamp=np.array(numpy_time_stamp)

            print('numpy_finger_pos_2 shape: ', numpy_finger_pos_2.shape) #  (3, 204446) in indy_20160407_02
            print('\n') 

            print('numpy_time_stamp: ',end='')
            print(numpy_time_stamp.shape)  #  (1, 204446)in indy_20160407_02

            sampling_rate=the_sampling_rate
            time_stamp_64ms=time_stamp[0][::sampling_rate]

            finger_z_pos_64ms=numpy_finger_pos_2[0][::sampling_rate]
            finger_x_pos_64ms=numpy_finger_pos_2[1][::sampling_rate]
            finger_y_pos_64ms=numpy_finger_pos_2[2][::sampling_rate]
            print('shape of finger_x_pos_64ms', finger_x_pos_64ms.shape) # (12778,) in indy_20160407_02

            finger_x_velocity=[]
            finger_y_velocity=[]
            finger_z_velocity=[]
            velocity_time_coor=[]

            finger_x_acceleration=[]
            finger_y_acceleration=[]
            finger_z_acceleration=[]
            acceleration_time_coor=[]

            duration=time_stamp_64ms.shape[0]

            for i in range(duration):
                #print('Velocity computing progress: ' + str( round( (i/duration)*100, 3) )+' %' )
                
                if ( i<duration-1 ):
                    #velocity=( finger_z_pos_64ms[i+1] - finger_z_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                    velocity=( finger_z_pos_64ms[i+1] - finger_z_pos_64ms[i] ) / ( 1 )
                    finger_z_velocity.append(velocity)

                    #velocity=( finger_x_pos_64ms[i+1] - finger_x_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                    velocity=( finger_x_pos_64ms[i+1] - finger_x_pos_64ms[i] ) / ( 1 )
                    finger_x_velocity.append(velocity)

                    #velocity=( finger_y_pos_64ms[i+1] - finger_y_pos_64ms[i] ) / ( time_stamp_64ms[i+1]-time_stamp_64ms[i] )
                    velocity=( finger_y_pos_64ms[i+1] - finger_y_pos_64ms[i] ) / ( 1 )
                    finger_y_velocity.append(velocity)

                    velocity_time_coor.append( numpy_time_stamp[0][i] )

                else:
                    '''
                    finger_x_velocity.append(0)
                    finger_y_velocity.append(0)
                    finger_z_velocity.append(0)
                    velocity_time_coor.append(0)
                    '''
                    pass
            
            finger_x_velocity=np.array(finger_x_velocity)
            finger_x_velocity=finger_x_velocity.astype(np.float32)
            
            finger_y_velocity=np.array(finger_y_velocity)
            finger_y_velocity=finger_y_velocity.astype(np.float32)

            finger_z_velocity=np.array(finger_z_velocity)
            finger_z_velocity=finger_z_velocity.astype(np.float32)

            velocity_time_coor=np.array(velocity_time_coor)

            duration=velocity_time_coor.shape[0]
            for i in range(duration):
                #print('Aceeleration computing progress '+ str( round( (i/duration)*100, 3) )+' %')

                if(i<duration-1):
                    #acceleration=(finger_x_velocity[i+1]-finger_x_velocity[i])/ (velocity_time_coor[i+1]-velocity_time_coor[i] )
                    acceleration=(finger_x_velocity[i+1]-finger_x_velocity[i])/ ( 1 )
                    finger_x_acceleration.append(acceleration)

                    #acceleration=(finger_y_velocity[i+1]-finger_y_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
                    acceleration=(finger_y_velocity[i+1]-finger_y_velocity[i])/( 1 )
                    finger_y_acceleration.append(acceleration)

                    #acceleration=(finger_z_velocity[i+1]-finger_z_velocity[i])/(velocity_time_coor[i+1]-velocity_time_coor[i])
                    acceleration=(finger_z_velocity[i+1]-finger_z_velocity[i])/( 1 )
                    finger_z_acceleration.append(acceleration)

                    acceleration_time_coor.append(velocity_time_coor[i])
                else:
                    '''
                    finger_x_acceleration.append(0)
                    finger_y_acceleration.append(0)
                    finger_z_acceleration.append(0)
                    acceleration_time_coor.append(0)
                    '''
                    pass
            finger_x_acceleration=np.array(finger_x_acceleration)
            finger_x_acceleration=finger_x_acceleration.astype(np.float32)
            
            finger_y_acceleration=np.array(finger_y_acceleration)
            finger_y_acceleration=finger_y_acceleration.astype(np.float32)
                
            finger_z_acceleration=np.array(finger_z_acceleration)
            finger_z_acceleration=finger_z_acceleration.astype(np.float32)

            acceleration_time_coor=np.array(acceleration_time_coor)

            x_velocity_label=finger_x_velocity
            y_velocity_label=finger_y_velocity
            z_velocity_label=finger_z_velocity

            x_acceleration_label=finger_x_acceleration
            y_acceleration_label=finger_y_acceleration
            z_acceleration_label=finger_z_acceleration
        return [x_velocity_label, y_velocity_label, z_velocity_label, x_acceleration_label, y_acceleration_label,  z_acceleration_label]