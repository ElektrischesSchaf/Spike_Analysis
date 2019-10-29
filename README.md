- [Spike_Analysis](#Spike-Analysis)  <br>
    - [Test](#Test-123-456)  <br>
    - [Electrodes](#Electrodes) <br>
    - [Plots](#plots)<br>
        - [Spike Train](#Spike-Train)<br> 
        - [Trajectory of finger tip](#Trajectory-of-finger-tip)<br>
        - [Velocity in each axis](#Velocity-in-each-axis)<br>
    - [Coefficient of Determination (R square) session indy_20160407_02](#Coefficient-of-Determination-R-square-session-indy_20160407_02)<br>
        - [With sorted spikes](#With-sorted-spikes)<br>
            - [With hash unit](#With-hash-unit)<br>
                - [R square of session indy_20160407_02, 0 time lag, 0-2 order, with sorted spikes](#r-square-of-session-indy_20160407_02-0-time-lag-0-2-order-with-sorted-spikes)  <br>
                - [R square of session indy_20160407_02, 1 time lag, 0-2 order, with sorted spikes](#r-square-of-session-indy_20160407_02-1-time-lag-0-2-order-with-sorted-spikes)  <br>
                - [R square of session indy_20160407_02, 2 time lag, 0-2 order, with sorted spikes](#r-square-of-session-indy_20160407_02-2-time-lag-0-2-order-with-sorted-spikes)  <br>
            - [Without hash unit](#Without-hash-unit)<br> 
                - [R square of session indy_20160407_02, 0 time lag, 0-2 order, with sorted spikes, without hash unit](#r-square-of-session-indy_20160407_02-0-time-lag-0-2-order-with-sorted-spikes-without-hash-unit)  <br>
                - [R square of session indy_20160407_02, 1 time lag, 0-2 order, with sorted spikes, without hash unit](#R-square-of-session-indy_20160407_02-1-time-lag-0-2-order-with-sorted-spikes-without-hash-unit) <br>
                - [R square of session indy_20160407_02, 2 time lag, 0-2 order, with sorted spikes, without hash unit](#R-square-of-session-indy_20160407_02-2-time-lag-0-2-order-with-sorted-spikes-without-hash-unit) <br>
        - [Without sorted spikes](#Without-sorted-spikes)  <br> <!-- TODO wrong answer -->
            - [With hash unit](#With-hash-unit)<br>
                - [R square of session indy_20160407_02, 0 time lag, 0-2 order, without sorted spikes](#r-square-of-session-indy_20160407_02-0-time-lag-0-2-order-without-sorted-spikes)  <br>
                - [R square of session indy_20160407_02, 1 time lag, 0-2 order, without sorted spikes](#r-square-of-session-indy_20160407_02-1-time-lag-0-2-order-without-sorted-spikes)  <br>
                - [R square of session indy_20160407_02, 2 time lag, 0-2 order, without sorted spikes](#r-square-of-session-indy_20160407_02-2-time-lag-0-2-order-without-sorted-spikes)  <br>
            - [Without hash unit](#Without-hash-unit)<br>  <!-- TODO wrong answer -->
                - [R square of session indy_20160407_02, 0 time lag, 0-2 order, without sorted spikes, without hash unit](#r-square-of-session-indy_20160407_02-0-time-lag-0-2-order-without-sorted-spikes-without-hash-unit) <br>
                - [R square of session indy_20160407_02, 1 time lag, 0-2 order, without sorted spikes, without hash unit](#r-square-of-session-indy_20160407_02-1-time-lag-0-2-order-without-sorted-spikes-without-hash-unit) <br> 
                - [R square of session indy_20160407_02, 2 time lag, 0-2 order, without sorted spikes, without hash unit](#r-square-of-session-indy_20160407_02-2-time-lag-0-2-order-without-sorted-spikes-without-hash-unit) <br>
    - [Coefficient of Determination (R square) in six sessions](#Coefficient-of-Determination-R-square-in-six-sessions) <br> <!-- TODO wrong hyper link -->
        - [With sorted spikes](#With-sorted-spikes)<br>
            - [With hash unit](#With-hash-unit)<br>
            - [Without hash unit](#Without-hash-unit)<br>
        - [Without sorted spikes](#Without-sorted-spikes)  <br>
            - [With hash unit](#With-hash-unit)<br>
            - [Without hash unit](#Without-hash-unit)<br>
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
![](/Spike_Train_Plots/Spike_Train_Channel_014.png)
* Spike trains from channel 14 for instance, has data from unit 1 to unit 3, ploted in different color.
---
### Trajectory of finger tip
![](/Kinematic_Variables_Plots/X-Y_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/X-Z_plane_trajectory.png)  ![](/Kinematic_Variables_Plots/Y-Z_plane_trajectory.png)  
* The trajectory of fingertips with all 204,446 data points.
---
### Velocity in each axis
![](/Kinematic_Variables_Plots/X_axis_velocity.png) ![](/Kinematic_Variables_Plots/Y_axis_velocity.png) ![](/Kinematic_Variables_Plots/Z_axis_velocity.png) 
* The velocity of fingertips in three axis with all 204,446 data points.
---

#### Test 123, 456
* This is a test for mark down.
---
## Coefficient of Determination (R square) session indy_20160407_02
### With sorted spikes
#### With hash unit
##### R square of session indy_20160407_02, 0 time lag, 0-2 order, with sorted spikes
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
##### R square of session indy_20160407_02, 1 time lag, 0-2 order, with sorted spikes
* model_x_position score:  0.12569332638109
* model_x_position_order_1 score:  0.19121025876136344
* model_x_position_order_2 score:  0.24131448913270948


* model_y_position score:  0.1883654397295187
* model_y_position_order_1 score:  0.27733616728240784
* model_y_position_order_2 score:  0.3259781350504727


* model_z_position score:  -0.020096794966500298


* model_x_velocity score:  0.38492102313746945
* model_x_velocity_order_1 score:  0.47289837901961285
* model_x_velocity_order_2 score:  0.5051083665868699


* model_y_velocity score:  0.39845265187567624
* model_y_velocity_order_1 score:  0.4967135656364531
* model_y_velocity_order_2 score:  0.5171511682883358


* model_z_velocity score: 0.01041646683870523


* model_x_acceleration score:  0.08492222487913426
* model_x_acceleration_order_1 score:  0.12244124635143983
* model_x_acceleration_order_2 score:  0.1321337729912596


* model_y_acceleration score:  0.07174566349791811
* model_y_acceleration_order_1 score:  0.09074300731256679
* model_y_acceleration_order_2 score:  0.08255187377340023


* model_z_acceleration score:  -0.024885176898776118
##### R square of session indy_20160407_02, 2 time lag, 0-2 order, with sorted spikes
* model_x_position score:  0.13867158926967982
* model_x_position_order_1 score:  0.20153751388691643
* model_x_position_order_2 score:  0.2516598304296219


* model_y_position score:  0.1886923280349605
* model_y_position_order_1 score:  0.2773548390536049
* model_y_position_order_2 score:  0.3224947705920258


* model_z_position score:  -0.02723079323106603


* model_x_velocity score:  0.4171485958566683
* model_x_velocity_order_1 score:  0.4648720030945891
* model_x_velocity_order_2 score:  0.4800379458182752


* model_y_velocity score:  0.4129862692451455
* model_y_velocity_order_1 score:  0.4674245344286795
* model_y_velocity_order_2 score:  0.4694093642653645


* model_z_velocity score: 0.02784815435220178


* model_x_acceleration score:  0.059299636801661615
* model_x_acceleration_order_1 score:  0.06281608763644664
* model_x_acceleration_order_2 score:  0.05369966995493425


* model_y_acceleration score:  0.0322030210034534
* model_y_acceleration_order_1 score:  0.036681044730094814
* model_y_acceleration_order_2 score:  0.013448340175180817


* model_z_acceleration score:  -0.013372545863992835
#### Without hash unit
##### R square of session indy_20160407_02, 0 time lag, 0-2 order, with sorted spikes, without hash unit
* model_x_position score:  0.05320797723806625
* model_x_position_order_1 score:  0.12293630416573931
* model_x_position_order_2 score:  0.16208585992536617


* model_y_position score:  0.13634050303606504
* model_y_position_order_1 score:  0.20914007080278063
* model_y_position_order_2 score:  0.2679578860986862


* model_z_position score:  -0.006950241036360438


* model_x_velocity score:  0.19633825208865352
* model_x_velocity_order_1 score:  0.2952228212653797
* model_x_velocity_order_2 score:  0.3546936420158616


* model_y_velocity score:  0.23640015547884263
* model_y_velocity_order_1 score:  0.35997823490496894
* model_y_velocity_order_2 score:  0.4293201745391175


* model_z_velocity score: -0.008900279782008358


* model_x_acceleration score:  0.0800968163147644
* model_x_acceleration_order_1 score:  0.10394188876250465
* model_x_acceleration_order_2 score:  0.1302247176484881


* model_y_acceleration score:  0.07166079612263421
* model_y_acceleration_order_1 score:  0.07574792153346999
* model_y_acceleration_order_2 score:  0.0838878683933727


* model_z_acceleration score:  -0.019265986133156554

##### R square of session indy_20160407_02, 1 time lag, 0-2 order, with sorted spikes, without hash unit
* model_x_position score:  0.05438623182345137
* model_x_position_order_1 score:  0.12985534983474212
* model_x_position_order_2 score:  0.17204655814405911


* model_y_position score:  0.13107654884997633
* model_y_position_order_1 score:  0.20272883674314368
* model_y_position_order_2 score:  0.2619760587943226


* model_z_position score:  -0.026081725145579382


* model_x_velocity score:  0.2519679704724773
* model_x_velocity_order_1 score:  0.34369440035395127
* model_x_velocity_order_2 score:  0.38156907457227474


* model_y_velocity score:  0.2988771718027967
* model_y_velocity_order_1 score:  0.40171097025802494
* model_y_velocity_order_2 score:  0.4397892239358572


* model_z_velocity score: 0.0029403883106152717


* model_x_acceleration score:  0.05668157759486325
* model_x_acceleration_order_1 score:  0.07098998200614604
* model_x_acceleration_order_2 score:  0.083770049532496


* model_y_acceleration score:  0.03546749309814767
* model_y_acceleration_order_1 score:  0.04081475561960113
* model_y_acceleration_order_2 score:  0.04184758847315606


* model_z_acceleration score:  -0.015901073706103608
##### R square of session indy_20160407_02, 2 time lag, 0-2 order, with sorted spikes, without hash unit
* model_x_position score:  0.06299533478119135
* model_x_position_order_1 score:  0.13981744928130235
* model_x_position_order_2 score:  0.18123005682339222


* model_y_position score:  0.12677812816366008
* model_y_position_order_1 score:  0.20044907909474974
* model_y_position_order_2 score:  0.25666654145411993


* model_z_position score:  -0.0333945234401658


* model_x_velocity score:  0.2921320029649911
* model_x_velocity_order_1 score:  0.35615754515255194
* model_x_velocity_order_2 score:  0.3774614300013849


* model_y_velocity score:  0.3257581493620284
* model_y_velocity_order_1 score:  0.38988093926157175
* model_y_velocity_order_2 score:  0.40865118452716565


* model_z_velocity score: 0.029110545771452556


* model_x_acceleration score:  0.02646952792945345
* model_x_acceleration_order_1 score:  0.026175647898692977
* model_x_acceleration_order_2 score:  0.020147844937716264


* model_y_acceleration score:  -0.0051596110675595774
* model_y_acceleration_order_1 score:  0.004252648734662023
* model_y_acceleration_order_2 score:  -0.006759090606800244


* model_z_acceleration score:  -0.0049404367137924066
### Without sorted spikes
#### With hash unit
##### R square of session indy_20160407_02, 0 time lag, 0-2 order, without sorted spikes
* model_x_position score in order  0 :  0.06810720356681921
* model_y_position score in order  0 :  0.15209316432387743
* model_z_position score in order  0 :  0.008055947915847184


* model_x_velocity score in order  0 :  0.2524634156611202
* model_y_velocity score in order  0 :  0.2648575655972646
* model_z_velocity score in order  0 :  0.026525974424062793


* model_x_acceleration score in order  0 :  0.07722587461617725
* model_y_acceleration score in order  0 :  0.08079761536891306
* model_z_acceleration score in order  0 :  0.00803567532624394
##### R square of session indy_20160407_02, 1 time lag, 0-2 order, without sorted spikes
* model_x_position score in order  0 :  0.03238224895620001
* model_y_position score in order  0 :  0.11372380933092374
* model_z_position score in order  0 :  0.00998565815796737


* model_x_velocity score in order  0 :  0.1806931888653328
* model_y_velocity score in order  0 :  0.1895093589840443
* model_z_velocity score in order  0 :  0.013542349317311242


* model_x_acceleration score in order  0 :  0.06416929272078209
* model_y_acceleration score in order  0 :  0.05991880232454305
* model_z_acceleration score in order  0 :  -0.006990135786655527

##### R square of session indy_20160407_02, 2 time lag, 0-2 order, without sorted spikes
* model_x_position score:  -0.027060138587140292
* model_x_position_order_1 score:  0.011881408298942353
* model_x_position_order_2 score:  0.047270576300165956


* model_y_position score:  0.031084111187886343
* model_y_position_order_1 score:  0.06562543703718382
* model_y_position_order_2 score:  0.09494998331821536


* model_z_position score:  -0.039260127910228704


* model_x_velocity score:  0.24980102776236235
* model_x_velocity_order_1 score:  0.30345341163575834
* model_x_velocity_order_2 score:  0.3315132648460495


* model_y_velocity score:  0.21741984135675507
* model_y_velocity_order_1 score:  0.2756812153949971
* model_y_velocity_order_2 score:  0.2949686441835412


* model_z_velocity score: -0.010791381546669365


model_x_acceleration score:  0.008785177220775209
model_x_acceleration_order_1 score:  0.018787426544658592
model_x_acceleration_order_2 score:  0.026905884027875082


model_y_acceleration score:  -0.002498051487009212
model_y_acceleration_order_1 score:  0.0022656953632115284
model_y_acceleration_order_2 score:  -0.017487995298690606


model_z_acceleration score:  -0.029691575320161423
#### Without hash unit
##### R square of session indy_20160407_02, 0 time lag, 0-2 order, without sorted spikes, without hash unit
* model_x_position score:  -0.05085609125259927
* model_x_position_order_1 score:  -0.015444630930580905
* model_x_position_order_2 score:  0.021141524998378625


* model_y_position score:  -0.01128414092087393
* model_y_position_order_1 score:  0.005747899596798112
* model_y_position_order_2 score:  0.023148010072180036


* model_z_position score:  -0.06578755229508526


* model_x_velocity score:  0.11077588788049808
* model_x_velocity_order_1 score:  0.16358906573918186
* model_x_velocity_order_2 score:  0.22823459526759016


* model_y_velocity score:  0.12791014489919272
* model_y_velocity_order_1 score:  0.18484510081172867
* model_y_velocity_order_2 score:  0.2326525950882119


* model_z_velocity score: -0.009478577143966938


* model_x_acceleration score:  0.036429703627523846
* model_x_acceleration_order_1 score:  0.05666104928863647
* model_x_acceleration_order_2 score:  0.054977248275219104


* model_y_acceleration score:  0.018712762597832056
* model_y_acceleration_order_1 score:  0.012468714600246678
* model_y_acceleration_order_2 score:  0.01080667399793378


* model_z_acceleration score:  -0.0063874712314186954
##### R square of session indy_20160407_02, 1 time lag, 0-2 order, without sorted spikes, without hash unit
* model_x_position score:  -0.04836601105549998
* model_x_position_order_1 score:  -0.008990075967148847
* model_x_position_order_2 score:  0.028388016112745884


* model_y_position score:  -0.012975784525332212
* model_y_position_order_1 score:  0.005381449025095186
* model_y_position_order_2 score:  0.025552618663644777


* model_z_position score:  -0.06948823628949241


* model_x_velocity score:  0.1484916080233507
* model_x_velocity_order_1 score:  0.20924094993900544
* model_x_velocity_order_2 score:  0.26087654972391994


* model_y_velocity score:  0.15714833194981515
* model_y_velocity_order_1 score:  0.21481675783389742
* model_y_velocity_order_2 score:  0.2478700108110652


* model_z_velocity score: 0.005063370562879088


* model_x_acceleration score:  0.02228843151214277
* model_x_acceleration_order_1 score:  0.02492359553884238
* model_x_acceleration_order_2 score:  0.032711735329897795


* model_y_acceleration score:  0.0035702478875359045
* model_y_acceleration_order_1 score:  -0.009882914788453379
* model_y_acceleration_order_2 score:  0.005632202727441582


* model_z_acceleration score:  0.006269897173626693

##### R square of session indy_20160407_02, 2 time lag, 0-2 order, without sorted spikes, without hash unit
* model_x_position score:  -0.045323976779240605
* model_x_position_order_1 score:  0.007364503546233414
* model_x_position_order_2 score:  0.039809356196933265


* model_y_position score:  -0.011076171421242176
* model_y_position_order_1 score:  0.0058402604624684384
* model_y_position_order_2 score:  0.031845166576678796


* model_z_position score:  -0.07848658010568199


* model_x_velocity score:  0.179914712147913
* model_x_velocity_order_1 score:  0.22690688773690226
* model_x_velocity_order_2 score:  0.2585282471549293


* model_y_velocity score:  0.1804827370495803
* model_y_velocity_order_1 score:  0.22463421372579595
* model_y_velocity_order_2 score:  0.24210745676880963


* model_z_velocity score: 0.013269039083495748


* model_x_acceleration score:  -0.005200576525754563
* model_x_acceleration_order_1 score:  0.006995990259295981
* model_x_acceleration_order_2 score:  0.01506527505214228


* model_y_acceleration score:  -0.01673706353781168
* model_y_acceleration_order_1 score:  -0.015617258616803742
* model_y_acceleration_order_2 score:  -0.006360472670094763


* model_z_acceleration score:  -0.010821068598025674

## Coefficient of Determination (R square) in six sessions
### With sorted spikes
#### With hash unit
* model_x_position score in order  0 :  0.21262676860054186
* model_y_position score in order  0 :  0.24682829773971315
* model_z_position score in order  0 :  0.389220262579711


* model_x_velocity score in order  0 :  0.24581712475512252
* model_y_velocity score in order  0 :  0.31679418367252543
* model_z_velocity score in order  0 :  0.05947447000796002


* model_x_acceleration score in order  0 :  0.11015972063649948
* model_y_acceleration score in order  0 :  0.11570917482294796
* model_z_acceleration score in order  0 :  0.008580461602417322
#### Without hash unit
* model_x_position score in order  0 :  0.15057048802839412
* model_y_position score in order  0 :  0.16293589088426463
* model_z_position score in order  0 :  0.33576073359282865


* model_x_velocity score in order  0 :  0.1673185007867367
* model_y_velocity score in order  0 :  0.18395478150869027
* model_z_velocity score in order  0 :  0.032943702596678226


* model_x_acceleration score in order  0 :  0.06687033858338154
* model_y_acceleration score in order  0 :  0.08346568473783367
* model_z_acceleration score in order  0 :  0.003008177941475565
### Without sorted spikes
#### With hash unit
* model_x_position score in order  0 :  0.18347082724949715
* model_y_position score in order  0 :  0.21237724001141434
* model_z_position score in order  0 :  0.3293247770818736


* model_x_velocity score in order  0 :  0.20294802123535383
* model_y_velocity score in order  0 :  0.29201350504515966
* model_z_velocity score in order  0 :  0.04815387599344323


* model_x_acceleration score in order  0 :  0.09907218649670813
* model_y_acceleration score in order  0 :  0.09051077809407226
* model_z_acceleration score in order  0 :  0.011354390220861266
#### Without hash unit
* model_x_position score in order  0 :  0.12904502531756812
* model_y_position score in order  0 :  0.13181480619340347
* model_z_position score in order  0 :  0.264018881740654


* model_x_velocity score in order  0 :  0.1382717089116846
* model_y_velocity score in order  0 :  0.15782791279045516
* model_z_velocity score in order  0 :  0.026107390222354088


* model_x_acceleration score in order  0 :  0.057761361819957724
* model_y_acceleration score in order  0 :  0.06273761046775095
* model_z_acceleration score in order  0 :  0.004607774874052617