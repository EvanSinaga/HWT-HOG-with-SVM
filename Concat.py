from ExtractHOG import haarhogfeat as hhft
from ExtractHOG import feature_path
import os
import numpy as np
from timeit import default_timer as timer

# cfile = "Concat.npy"
# cfile = "Concat_ND.npy" # No diagonal
# cfile = "Concat_NLL.npy" # No LL
cfile = "Concat_LL.npy" # LL
cfile_path = os.path.join(feature_path, cfile)

print("Start Concating ...")
start = timer()
# Xhog = np.array([list(np.concatenate(coeff[1:]).flat) for i, coeff in enumerate(hhft)])
Xhog = np.array([list(coeff[0]) for i, coeff in enumerate(hhft)]) # LL
print("\nFinish Concating ...")
print("\n[INFO] Saving {} ...".format(cfile))
np.save(cfile_path, Xhog)
print("[INFO] {} saved".format(cfile))
print("Concatted shape: ", Xhog.shape)
end = timer()
print(end - start)
print("Debug")