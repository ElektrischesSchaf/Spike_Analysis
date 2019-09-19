# Spike_Analysis
### Spikes are signals generated from the single frequency and have magnitude significantly larger than noise, the voltage drop from neural soma and axon membrane. Spike train is time-series data which comes from a neuron. In this repository I use the mat file "indy_20160407_02.mat" downloaded from [Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology](https://zenodo.org/record/583331#.XWirEigzZPb). This dataset has 96 channels and each channel contains 1-6 units.  

![](/Spike_Train_Plots/Spike_Train_Channel_14.png)
#### Spike trains from channel 14 for instance, has data from unit 1 to unit 3, ploted in different color.
---
![](/Kinematic_Variables_Plots/X-Y_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/X-Z_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/Y-Z_plane_trajectory.png)  
#### The trajectory of fingertips with all 204,446 data points.
---
![](/Kinematic_Variables_Plots/X_axis_velocity.png) ![](/Kinematic_Variables_Plots/Y_axis_velocity.png) ![](/Kinematic_Variables_Plots/Z_axis_velocity.png) 
#### The velocity of fingertips in three axis with all 204,446 data points.
---
#### R\sqrt{2} of 1 session, no time lag, zero order 
* model_x_position score: 0.1035952123342534
* model_y_position score: 0.18687209392892745
* model_z_position score: -12847.714342367937
* model_x_velocity score: 0.2902760117538522
* model_y_velocity score: 0.32194629832341504
* model_z_velocity score: 0.0018726782972583456
* model_x_acceleration score: 0.10033632067101772
* model_y_acceleration score: 0.09852761885686134
* model_z_acceleration score: -0.02405328773631754
