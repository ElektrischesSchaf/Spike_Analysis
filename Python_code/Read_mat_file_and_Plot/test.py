import numpy as np

aArray=np.array([1,1,1])

bArray=np.array([2,2,2])

aList=[aArray, bArray]

print('aList length: ',end='')
print( len(aList) )
xArray=np.array(aList)
print('xArray shape: ',end='')
print(xArray.shape)

BList=[[]]

BList[-1].append(1)
BList[-1].append(1)
BList[-1].append(1)
BList.append([])
BList[-1].append(2)
BList[-1].append(2)
BList[-1].append(2)

BList.append([3, 3, 3])

print('BList length: ',end='')
print(len(BList))

BArray=np.array(BList)

print('BArray shape: ',end='')
print(BArray.shape)

C_array=np.array([1,2,3,4,5])
print('C_array shape: ',end='')
print(C_array.shape)
print(C_array[2])