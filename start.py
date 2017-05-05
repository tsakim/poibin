import poibin
import numpy as np
p = np.random.random_sample(size=5000)
pb = poibin.PoiBin(p)
pb.pmf([0, 1, 2, 3])
