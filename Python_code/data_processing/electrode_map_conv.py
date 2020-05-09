import numpy as np

class map_conv_2D():
    def __init__(self, kernel_size, instance_phase_all_channels):
        self.indy_M1_electrode_map=np.array( [
            [0, 42, 46, 25, 31, 35, 39, 41, 47, 0], 
            [38, 40, 48, 27, 29, 33, 37, 43, 6, 45], 
            [34, 36, 44, 1, 9, 13, 17, 21, 2, 88], 
            [30, 32, 89, 93, 5, 15, 19, 23, 8, 84], 
            [26, 28, 81, 85, 87, 91, 7, 4, 86, 80], 
            [22, 24, 77, 79, 83, 3, 11, 66, 82, 76], 
            [18, 20, 73, 75, 95, 54, 62, 74, 78, 72], 
            [14, 16, 94, 96, 57, 58, 50, 70, 64, 68], 
            [10, 12, 90, 92, 61, 65, 69, 71, 56, 60], 
            [0, 51, 49, 53, 55, 59, 63, 67, 52, 0]
        ], np.int32)
        self.kernel_size=kernel_size
        self.instance_phase_all_channels=instance_phase_all_channels

    def conv2d_phase_clustering(self):
        ITPC_angle_output = np.empty([self.instance_phase_all_channels.shape[1], 0])
        ITPC_abs_output = np.empty([self.instance_phase_all_channels.shape[1], 0])

        for x in range( int(self.indy_M1_electrode_map.shape[0]) - self.kernel_size + 1):
            for y in range( int(self.indy_M1_electrode_map.shape[1]) - self.kernel_size + 1):
                sub_instance_phase_all_channels=np.empty([ 0, self.instance_phase_all_channels.shape[1] ])
                # sub_instance_phase_all_channels=np.empty([0])
                sub_map=self.indy_M1_electrode_map[ x:x+self.kernel_size , y:y+self.kernel_size ]
                sub_map=sub_map.flatten()
                sub_map=list(sub_map)
                # print(sub_map)

                for ele in sub_map:
                    if ele != 0: # channel index start from 0
                        ele=ele-1
                        new=np.reshape(self.instance_phase_all_channels[ele, :], (1,-1) )
                        sub_instance_phase_all_channels=np.concatenate(( sub_instance_phase_all_channels, new ) , axis=0)                        

                ITPC_angle=[]
                ITPC_abs=[]
                for itpc_loop in range( sub_instance_phase_all_channels.shape[1] ) :
                    itpc_angle=0
                    itpc_abs=0
                    itpc_angle=np.angle( np.mean (np.exp( 1j * sub_instance_phase_all_channels[:,itpc_loop]  )))
                    itpc_abs=np.abs( np.mean (np.exp( 1j * sub_instance_phase_all_channels[:,itpc_loop]  )))

                    ITPC_angle.append(itpc_angle)
                    ITPC_abs.append(itpc_abs)

                ITPC_angle=np.array(ITPC_angle).transpose()
                ITPC_abs=np.array(ITPC_abs).transpose()

                print('\nITPC_angle_output= ', ITPC_angle_output.shape)
                ITPC_angle=np.reshape(ITPC_angle, (ITPC_angle.shape[0],-1) )
                # print('\nITPC_angle= ', ITPC_angle.shape)
                print('\nITPC_abs_output= ', ITPC_abs_output.shape)
                ITPC_abs=np.reshape(ITPC_abs, (ITPC_abs.shape[0],-1) )
                # print('\nITPC_abs= ', ITPC_abs.shape)
            
                ITPC_angle_output=np.concatenate( ( ITPC_angle_output, ITPC_angle), axis=1)
                ITPC_abs_output=np.concatenate( ( ITPC_abs_output, ITPC_abs), axis=1)

                # print('finised a sub map\n')

        # return aveage_phase, synchronicity
        return ITPC_angle_output, ITPC_abs_output