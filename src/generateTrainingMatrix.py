#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt


from scipy.signal import resample
from scipy.spatial.distance import euclidean

from armada18eeg import *


"""
Does what the program is expected to do, assuming the arguments have been parsed.
directory_path: The directory where the script will look for the files to process.
output_file: The filename of the generated output file.
"""
def do_the_job(directory_path, output_file):
	for (dirpath, dirnames, filenames) in os.walk(directory_path):
		for x in filenames:
			# We will be ignoring files not ending in .csv
			if not x.lower().endswith('.csv'):
				continue
			# We will be ignoring files containing the substring "test" (TEST FILES SHOULD NOT BE IN THE DATASET DIRECTORY IN THE FIRST PLACE)
			if 'test' in x.lower():
				continue
			try:
				name, state, _ = x[:-4].split('-')
			except:
				print ('Wrong file name', x)
				sys.exit(-1)
			if state.lower() == 'concentrating':
				state = 2.
			elif state.lower() == 'neutral':
				state = 1.
			elif state.lower() == 'relaxed':
				state = 0.
			else:
				print ('Wrong file name', x)
				sys.exit(-1)
			full_file_path = dirpath  +   '/'   + x
			print ('Using file', x)
			vectors, header = generate_feature_vectors_from_samples(full_file_path, samples=150, period=1., state=state)
			print ('resulting vector shape for the file', vectors.shape)
			try:
				FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )
			except UnboundLocalError:
				FINAL_MATRIX = vectors

	print ('FINAL_MATRIX', FINAL_MATRIX.shape)
	np.random.shuffle(FINAL_MATRIX)
	np.savetxt(output_file, FINAL_MATRIX, delimiter=',', header=header)


"""
Main function. The parameters for the script are the following:
[1] directory_path: The directory where the script will look for the files to process.
[2] output_file: The filename of the generated output file.
"""
if __name__ == '__main__':
	if len(sys.argv) <3:
		print ('arg1: input dir\narg2: output file')
		sys.exit(-1)
	directory_path = sys.argv[1]
	output_file = sys.argv[2]
	do_the_job(directory_path, output_file)
