from Helper import convertColor
from ExtractHaar import haar_feat
from ExtractHOG import haarhogfeat, haarhogvisual
from Undersample import X_resampled

import numpy as np
import matplotlib.pyplot as plt

haar = haar_feat
hogft, hogvs = haarhogfeat, haarhogvisual

print(haar.shape) # (856, 4, 128, 128)
print(type(haar))
print(hogft.shape) # (856, 4, 8100)
print(hogvs.shape) # (856, 4, 128, 128)




LL = haar[700][0]
plt.imshow(LL, cmap='gray')
plt.show()

HL = haar[700][1]
plt.imshow(HL, cmap='gray')
plt.show()

LH = haar[700][2]
plt.imshow(LH, cmap='gray')
plt.show()

HH = haar[700][3]
plt.imshow(HH, cmap='gray')
plt.show()

# hog = hogvs[700][0]
# plt.imshow(hog, cmap='gray')
# plt.show()

# print("Debug")

