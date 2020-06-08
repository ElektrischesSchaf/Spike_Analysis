import torch
a=torch.tensor([
    [1,1,1,2,2,2,3,3,3,11,22,33],
    [4,4,4,5,5,5,6,6,6,44,55,66],
    [7,7,7,8,8,8,9,9,9,77,88,99],
])
print('shape of a= ', a.size())

b=a.clone()
b=b.view( a.size(0), -1, 3 )
print('\nshape of b= ', b.size())
print(b)