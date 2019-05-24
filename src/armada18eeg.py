#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np


from scipy.signal import resample
from scipy.spatial.distance import euclidean
from scipy.stats import moment, entropy
from scipy.linalg import logm
from scipy.fftpack import fft


"""
Returns the data matrix given the path of a CSV file.
"""
def matrix_from_csv_file(file_path):
	csv_data = np.genfromtxt(file_path, delimiter=',')
	matrix = csv_data[1:]
	headers = csv_data[0]
	#print 'CSV', (csv_data.shape)
	#print 'MAT', (matrix.shape)
	#print 'HDR', (headers.shape)
	return matrix


"""
Returns a slice of the given matrix, where start is the offset and period is used to specify the length of the signal.
"""
def get_time_slice(matrix, start=0, period=1.):
	rstart = matrix[0, 0] + start
	end   = rstart + period

	index_0 = 0
	max_index = matrix.shape[0]
	
	while matrix[index_0,0] < rstart:
		index_0 += 1

	index_1 = index_0
	while matrix[index_1,0] < end:
		index_1 += 1

	return matrix[index_0:index_1, :], matrix[index_1,0]-matrix[index_0,0]


"""
Returns the average value of the signals.
"""
def feature_mean(matrix):
	ret = np.mean(matrix, axis=0).flatten()
	names = ['mean_'+str(i) for i in range(len(ret))]
	return ret, names

"""
Returns the derivate of the average value of the signals.
"""
def feature_mean_d(h1, h2):
	ret = (feature_mean(h2)[0]-feature_mean(h1)[0]).flatten()
	names = ['mean_d_'+str(i) for i in range(len(ret))]
	return ret, names

"""
Returns the average values of the four quarters signals.
"""
def feature_mean_q(q1, q2, q3, q4):
	v1 = feature_mean(q1)[0]
	v2 = feature_mean(q2)[0]
	v3 = feature_mean(q3)[0]
	v4 = feature_mean(q4)[0]
	ret = np.hstack([ v1, v2, v3, v4, v1-v2, v1-v3, v1-v4, v2-v3, v2-v4, v3-v4 ]).flatten()
	names = ['mean_d_'+str(i) for i in range(len(ret))]
	return ret, names

"""
Returns the standard deviation of a signal matrix.
"""
def feature_stddev(matrix):
	ret = np.std(matrix, axis=0).flatten()
	names = ['stddev_'+str(i) for i in range(len(ret))]
	return ret, names



"""
Returns the derivate of the standard deviation of a signal matrix.
"""
def feature_stddev_d(h1, h2):
	ret = (feature_stddev(h2)[0]-feature_stddev(h1)[0]).flatten()
	names = ['stddev_d_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the statistical NORMALISED moments of a set of signals.
Note that despite that the 1st and 2nd raw moments are the mean and variance of the signal, normalised moments have fixed values of 0 and 1, respectively.
Therefore, this function returns the 3rd, 4th, 5th and 6th moments.
"""
def feature_moments(matrix):
	ret = moment(matrix, moment=[3,4,5,6], axis=0, nan_policy='propagate').flatten()
	names = ['moments_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the maximum values of the signals.
"""
def feature_max(matrix):
	ret = np.max(matrix, axis=0).flatten()
	names = ['max_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the derivate of the maximum values of the signals.
"""
def feature_max_d(h1, h2):
	ret = (feature_max(h2)[0]-feature_max(h1)[0]).flatten()
	names = ['max_d_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the maximum values of the four quarters signals.
"""
def feature_max_q(q1, q2, q3, q4):
	v1 = feature_max(q1)[0]
	v2 = feature_max(q2)[0]
	v3 = feature_max(q3)[0]
	v4 = feature_max(q4)[0]
	ret = np.hstack([ v1, v2, v3, v4, v1-v2, v1-v3, v1-v4, v2-v3, v2-v4, v3-v4 ]).flatten()
	names = ['max_q_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the minimum values of the signals.
"""
def feature_min(matrix):
	ret = np.min(matrix, axis=0).flatten()
	names = ['min_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the derivate of the minimum values of the signals.
"""
def feature_min_d(h1, h2):
	ret = (feature_min(h2)[0]-feature_min(h1)[0]).flatten()
	names = ['min_d_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the minimum values of the four quarters signals.
"""
def feature_min_q(q1, q2, q3, q4):
	v1 = feature_min(q1)[0]
	v2 = feature_min(q2)[0]
	v3 = feature_min(q3)[0]
	v4 = feature_min(q4)[0]
	ret = np.hstack([ v1, v2, v3, v4, v1-v2, v1-v3, v1-v4, v2-v3, v2-v4, v3-v4 ]).flatten()
	names = ['min_q_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Covariance matrix-based feature. NOTE THAT WE ARE DISCARDING PART OF THE DATA FROM 'f_mean_q' to get a square matrix
"""
def feature_covariance_matrix(f_mean_q, f_max_q, f_min_q):
	# f_mean_q contains the mean of the four parts of the signals (v1, v2, v3, v4... 4 total) plus the differences
	# between them (v1-v2, v1-v3, v1-v4, v2-v3, v2-v4, v3-v4... 6 total) for each signal. However, for things to add up,
	# we are discarding the six last elements of the vector, that is, the differences for the last of the signals.
	# That leaves us with 10*5-6 elements: 10*5 from all the values, -6 for the values we are discarding.
	# f_max_q and f_min_q work similarly as f_mean_q but with the maximum and minimum values, respectively. They are are
	# both 10*5 matrices and we use them all. The resulting number of elements is 144, whose square root is 12.

	# USING_SHRINK = False
	USING_SHRINK = True

	size = float(f_mean_q.size + f_max_q.size + f_min_q.size)
	square = np.sqrt(size)
	pad = np.array([])
	if int(square + 0.5) ** 2 == size:
		remove = 0
	elif USING_SHRINK:
		shrink_size = int(square) ** 2
		remove = int(size - shrink_size)
	else:
		padded_size = int(square+1.) ** 2
		pad = np.zeros((int(padded_size - size)))
		remove = 0

	if remove == 0:
		src_matrix = np.hstack([ f_mean_q, pad, f_max_q, f_min_q ])
	else:
		src_matrix = np.hstack([ f_mean_q[:-remove], f_max_q, f_min_q ])
	s = int(np.sqrt(src_matrix.size))
	src_matrix = src_matrix.reshape((s,s))
	cov_matrix = np.cov(src_matrix)
	flattened = cov_matrix.flatten()
	names = ['covmat_'+str(i) for i in range(len(flattened))]
	return flattened, cov_matrix, names


def feature_eigenvalues(cov_matrix):
	"""
	Returns the eigenvalues of the covariance matrix (see feature_covariance_matrix).
	"""
	ret = np.linalg.eigvals(cov_matrix).flatten()
	names = ['eigen_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the log matrix.
"""
def feature_logm(cov_matrix):
	m = logm(cov_matrix)
	r = np.hstack([ np.diagonal(m, i) for i in range(m.shape[0]) ])
	ret = np.real(r.flatten())
	names = ['logm_'+str(i) for i in range(len(ret))]
	return ret, names


"""
Returns the entropy
"""
def feature_entropy(matrix):
	ret = entropy(matrix).flatten()
	names = ['entropy'+str(i) for i in range(len(ret))]
	return ret, names

def	feature_correlate(h1, h2):
	"""
	Returns the correlation between the first and second part of a signal.
	"""
	ret = np.array([np.correlate(h1[i], h2[i]) for i in range(h1.shape[0])]).flatten()
	names = ['correlate_'+str(i) for i in range(len(ret))]
	return ret, names


def feature_fft(matrix):
	"""
	Returns the FFT of the signals.
	"""
	ret = np.real(fft(matrix).flatten())
	names = ['fft_'+str(i) for i in range(len(ret))]
	return ret, names




def feature_vector(matrix, state):
	"""
	Uses the previously defined functions to compute and return all the features considered.
	"""
	# Compute here the two parts
	h1, h2 = np.split(matrix, [ int(matrix.shape[0]/2) ])
	q1, q2, q3, q4 = np.split(matrix, [int(0.25*matrix.shape[0]), int(0.5*matrix.shape[0]), int(0.75*matrix.shape[0])  ])

	variables = []

	# Mean
	f_mean, v = feature_mean(matrix)
	variables += v

	# Derivate of the mean 
	f_mean_d, v = feature_mean_d(h1, h2)
	variables += v

	# Mean-Q
	f_mean_q, v = feature_mean_q(q1, q2, q3, q4)
	variables += v

	# Standard deviation
	f_stddev, v = feature_stddev(matrix)
	variables += v

	# Standard deviation
	f_stddev_d, v = feature_stddev_d(h1, h2)
	variables += v

	# Moments (3rd, 4th, 5th, 6th)
	f_moments, v = feature_moments(matrix)
	variables += v

	# Maximum value
	f_max, v = feature_max(matrix)
	variables += v

	# Derivate of the maximum value
	f_max_d, v = feature_max_d(h1, h2)
	variables += v

	# Max-Q
	f_max_q, v = feature_max_q(q1, q2, q3, q4)
	variables += v

	# Minimum value
	f_min, v = feature_min(matrix)
	variables += v

	# Derivate of the minimum value
	f_min_d, v = feature_min_d(h1, h2)
	variables += v

	# Min-Q
	f_min_q, v = feature_min_q(q1, q2, q3, q4)
	variables += v

	# Covariance matrix
	f_covariance_matrix, covariance_matrix, v = feature_covariance_matrix(f_mean_q, f_max_q, f_min_q)
	variables += v

	# Eigenvalues of the covariance matrix
	f_eigenvalues, v = feature_eigenvalues(covariance_matrix)
	variables += v

	# Upper triangle of the logm matrix of the covariance matrix
	f_logm, v = feature_logm(covariance_matrix)
	variables += v

	# Energy of entropy
	f_entropy, v = feature_entropy(matrix)
	variables += v

	# Correlation
	f_correlate, v = feature_correlate(h1, h2)
	variables += v

	# FFT & power spectrum
	f_fft, v = feature_fft(matrix)
	variables += v

	if state == None:
		ret = np.hstack([f_mean, f_mean_d, f_mean_q, f_stddev, f_stddev_d, f_moments, f_max, f_max_d, f_max_q, f_min, f_min_d, f_min_q, f_covariance_matrix, f_eigenvalues, f_logm, f_entropy, f_correlate, f_fft])
	else:
		ret = np.hstack([f_mean, f_mean_d, f_mean_q, f_stddev, f_stddev_d, f_moments, f_max, f_max_d, f_max_q, f_min, f_min_d, f_min_q, f_covariance_matrix, f_eigenvalues, f_logm, f_entropy, f_correlate, f_fft, np.array([state]) ])
	return ret, variables


def generate_feature_vectors_from_samples(full_file_path, samples, period, state=None):
	"""
	Returns a number of feature vectors and a CSV header corresponding to the features generated from a labeled CSV file.
	full_file_path: The path of the file to be read
	samples: size of the re-sampled vector
	period: period of the time used to compute feature vectors
	state: label for the feature vector
	"""
	# Read the matrix from file
	matrix = matrix_from_csv_file(full_file_path)
	# We will start at the very beginning of the file
	t = 0.
	# No previous vector is available at the start
	previous_vector = None
	# Until an exception is raised or a stop condition is met
	while True:
		# Get the next slice from the file (starting at time 't', with a duration of 'period'
		# If an exception is raised or the slice is not as long as we expected, return the
		# current data available
		try:
			s, ts = get_time_slice(matrix, t, period)
		except IndexError:
			break
		if len(s) == 0:
			break
		if ts < 0.9 * period:
			break
		# Perform the resampling of the vector
		ry, rx = resample(s[:, 1:], samples, t=s[:, 0], axis=0)
		# Slide the slice
		t+=0.5*period
		# Compute the feature vector. We will be appending two of these. Therefore, if there was
		# no previous vector we set it and continue with the next vector
		r, headers = feature_vector(ry, state)
		if previous_vector is None:
			previous_vector = r[:-1] # keep in mind that we remove the label (last column) from the previous vector
			continue
		# If there was a previous vector, the script concatenate the two vectors and add the result
		else:
			final_concat_vector = np.hstack([previous_vector, r])
			try:
				##print final_concat_vector.shape
				ret = np.vstack([ret, final_concat_vector])
			except UnboundLocalError:
				ret = final_concat_vector
			previous_vector = r[:-1] # keep in mind that we remove the label (last column) from the previous vector
	# Return all the vectors computed
	header_a = ','.join([x+'_a' for x in headers])
	header_b = ','.join([x+'_b' for x in headers])
	return ret, header_a+','+header_b+',label'

