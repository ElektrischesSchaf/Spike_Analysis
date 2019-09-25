- [Spike_Analysis](#spike-analysis)
      - [Spike trains from channel 14 for instance, has data from unit 1 to unit 3, ploted in different color.](#spike-trains-from-channel-14-for-instance--has-data-from-unit-1-to-unit-3--ploted-in-different-color)  
      - [The trajectory of fingertips with all 204,446 data points.](#the-trajectory-of-fingertips-with-all-204-446-data-points)  
      - [The velocity of fingertips in three axis with all 204,446 data points.](#the-velocity-of-fingertips-in-three-axis-with-all-204-446-data-points)
      - [R square of 1 session, no time lag, zero order](#r-square-of-1-session--no-time-lag--zero-order)
      - [R square of 1 session, no time lag, zero order](#r-square-of-1-session--no-time-lag--zero-order-1)

# Spike_Analysis
* Spikes are signals generated from the single frequency and have magnitude significantly larger than noise, the voltage drop from neural soma and axon membrane. Spike train is time-series data which comes from a neuron. In this repository I use the mat file "indy_20160407_02.mat" downloaded from [Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology](https://zenodo.org/record/583331#.XWirEigzZPb). This dataset has 96 channels and each channel contains 1-6 units.  



![](/Spike_Train_Plots/Spike_Train_Channel_14.png)
#### Spike trains from channel 14 for instance, has data from unit 1 to unit 3, ploted in different color.
---
![](/Kinematic_Variables_Plots/X-Y_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/X-Z_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/Y-Z_plane_trajectory.png)  
#### The trajectory of fingertips with all 204,446 data points.
---
![](/Kinematic_Variables_Plots/X_axis_velocity.png) ![](/Kinematic_Variables_Plots/Y_axis_velocity.png) ![](/Kinematic_Variables_Plots/Z_axis_velocity.png) 
#### The velocity of fingertips in three axis with all 204,446 data points.
---
#### R square of 1 session, no time lag, zero order
* model_x_position score:  0.11435610816030484
* model_x_position_order_1 score:  0.1793229330330831
* model_x_position_order_2 score:  0.2297033077459666


* model_y_position score:  0.18804542863076756
* model_y_position_order_1 score:  0.27787899489975054
* model_y_position_order_2 score:  0.32723045669753703


* model_z_position score:  -0.010893968477762472


* model_x_velocity score:  0.2908923622211801
* model_x_velocity_order_1 score:  0.4121560188809077
* model_x_velocity_order_2 score:  0.4866755385777508


* model_y_velocity score:  0.3213251108746724
* model_y_velocity_order_1 score:  0.4540839937501209
* model_y_velocity_order_2 score:  0.5162227548532502


* model_z_velocity score: -0.0007922085518157207


* model_x_acceleration score:  0.09966378444822588
* model_x_acceleration_order_1 score:  0.1410982611860867
* model_x_acceleration_order_2 score:  0.18135056353669943


* model_y_acceleration score:  0.0975972421152016
* model_y_acceleration_order_1 score:  0.127385234152242
* model_y_acceleration_order_2 score:  0.15411981035507605
---
#### R square of 1 session, no time lag, zero order
