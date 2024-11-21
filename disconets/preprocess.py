# %%
import numpy as np
import time
import sys
###################### ADD PATH TO THE DEEP PRIOR PACKAGE HERE
sys.path.append('deep-prior/src/')
sys.path.append('DISCONets')

# %%
from data.dataset import NYUDataset
from data.importers import NYUImporter

# %%
di = NYUImporter('./dataset/')

# %%
Seq = di.loadSequence('train')
trainDataset = NYUDataset([Seq])
X_train, Y_train = trainDataset.imgStackDepthOnly('train')
np.save('data/X_train', X_train)
np.save('data/Y_train', Y_train)

Seq = di.loadSequence('test')
testDataset = NYUDataset([Seq])
X_test, Y_test = testDataset.imgStackDepthOnly('test')
np.save('data/X_test', X_test)
np.save('data/Y_test', Y_test)