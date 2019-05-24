#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import resample
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from armada18eeg import *



# We read two files (the second one is actually the same as the first one, slightly modified.
# The first column is populated with the timestamps, the rest are the signals. 
matrix1 = matrix_from_csv_file(sys.argv[1])
matrix2 = matrix_from_csv_file(sys.argv[2])

# We are going to select a particular time slice from the two files and plot it in bright colors.
s1 = get_time_slice(matrix1, 0.0, 0.2)
s2 = get_time_slice(matrix2, 0.0, 0.2)
plt.subplot(2, 2, 1)
plt.plot(s1[:, 0], s1[:, 1:], color='#ff0000')
plt.subplot(2, 2, 2)
plt.plot(s2[:, 0], s2[:, 1:], color='#0000ff')

# We print their original shapes.
print 'Original slice 1', s1.shape
print 'Original slice 2', s2.shape



# Then, we resample the signals to 100 measures, separating the time and the signals.
rsy1, rsx1 = resample(s1[:, 1:], 75, t=s1[:, 0], axis=0)
rsy2, rsx2 = resample(s2[:, 1:], 75, t=s2[:, 0], axis=0)




#print 'Mean 1', feature_mean(rsy1)
#print 'Moments 1', feature_moments(rsy1)
#print 'Mean 1', feature_mean(rsy2)
#print 'Moments 2', feature_moments(rsy2)
#print feature_mean(rsy1).shape, feature_moments(rsy1).shape

whole1 =  feature_vector(rsy1)
print 'Whole 1', whole1
print '--------------'




# We print now the resampled signals
print 'Resampled 1', rsx1.shape, rsy1.shape
print 'Resampled 2', rsx2.shape, rsy2.shape
plt.subplot(2, 2, 3)
plt.plot(rsx1, rsy1, color='#880000')
plt.subplot(2, 2, 4)
plt.plot(rsx2, rsy2, color='#000088')


plt.ylabel('mv')
plt.show()








