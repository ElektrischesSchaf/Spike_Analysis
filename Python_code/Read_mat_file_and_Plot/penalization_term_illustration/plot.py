from scipy.special import softmax
import numpy as np
import pandas as pd

x = np.random.rand(10,20)
A = softmax(x, axis=1)

df = pd.DataFrame( A )
df.to_csv( 'A.csv', index=False, header=False)

A_transpose = np.transpose(A)

df = pd.DataFrame( A_transpose )
df.to_csv( 'A_transpose.csv', index=False, header=False)

my_identity_matrix = np.eye( A.shape[0] , dtype=float )

df = pd.DataFrame( my_identity_matrix )
df.to_csv( 'identity_matrix.csv', index=False, header=False)

result = np.matmul(A,A_transpose) - my_identity_matrix

df = pd.DataFrame( result )
df.to_csv( 'result.csv', index=False, header=False)