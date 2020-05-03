- [Spike_Analysis](#Spike-Analysis)  <br>
    - [Test](#Test-123-456)  <br>
    - [Electrodes](#Electrodes) <br>
    - [Plots](#plots)<br>
        - [Spike Train](#Spike-Train)<br> 
        - [Trajectory of finger tip](#Trajectory-of-finger-tip)<br>
        - [Velocity in each axis](#Velocity-in-each-axis)<br>

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Spike Analysis
* Spikes are signals generated from the single frequency and have magnitude significantly larger than noise, the voltage drop from neural soma and axon membrane. Spike train is time-series data which comes from a neuron. In this repository I use the mat file "indy_20160407_02.mat" downloaded from [Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology](https://zenodo.org/record/583331#.XWirEigzZPb). This dataset has 96 channels and each channel contains 1-6 units.  

## Electrodes
* Indy M1 <br>
	NaN    42    46    25    31    35    39    41    47   NaN <br>
	38    40    48    27    29    33    37    43     6    45 <br>
	34    36    44     1     9    13    17    21     2    88 <br>
	30    32    89    93     5    15    19    23     8    84 <br>
	26    28    81    85    87    91     7     4    86    80 <br>
	22    24    77    79    83     3    11    66    82    76 <br>
	18    20    73    75    95    54    62    74    78    72 <br>
	14    16    94    96    57    58    50    70    64    68 <br>
	10    12    90    92    61    65    69    71    56    60 <br>
	NaN    51    49    53    55    59    63    67    52   NaN <br>

* Indy S1 <br>
	38    42    46    25   NaN    35    39    41    47    45 <br>
	NaN    40    48    27    29    33    37    43     6   NaN <br>
	34    36    44     1     9    13    17    21     2    88 <br>
	30    32    89    93     5    15    19    23     8    84 <br>
	26    28    81    85    87    91     7     4    86    80 <br>
	22    24    77    79    83     3    11    66    82    76 <br>
	18    20    73    75    95    54    62    74    78    72 <br>
	14    16    94    96    57    58    50    70    64    68 <br>
	10    12    90    92    61    65    69    71    56    60 <br>
	NaN    51    49    53    55    59    63    67    52    31 <br>

* Loco M1 and S1  <br>
	NaN    42    46    25    31    35    39    41    47   NaN <br>
	38    40    48    27    29    33    37    43     6    45 <br>
	34    36    44     1     9    13    17    21     2    88 <br>
	30    32    89    93     5    15    19    23     8    84 <br>
	26    28    81    85    87    91     7     4    86    80 <br>
	22    24    77    79    83     3    11    66    82    76 <br>
	18    20    73    75    95    54    62    74    78    72 <br>
	14    16    94    96    57    58    50    70    64    68 <br>
	10    12    90    92    61    65    69    71    56    60 <br>
	NaN    51    49    53    55    59    63    67    52   NaN <br>

## Plots
### Spike Train
![](/Figures/Spike_Train_Plots/Spike_Train_Channel_014.png)
* Spike trains from channel 14 for instance, has data from unit 1 to unit 3, ploted in different color.
---
### Trajectory of finger tip
![](/Figures/Kinematic_Variables_Plots/X-Y_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/X-Z_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/Y-Z_plane_trajectory.png)  
* The trajectory of fingertips with all 204,446 data points.
---
### Velocity in each axis
![](/Figures/Kinematic_Variables_Plots/X_axis_velocity.png) ![](/Kinematic_Variables_Plots/Y_axis_velocity.png) ![](/Kinematic_Variables_Plots/Z_axis_velocity.png) 
* The velocity of fingertips in three axis with all 204,446 data points.
---