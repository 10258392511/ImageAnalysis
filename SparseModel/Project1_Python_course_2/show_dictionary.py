import numpy as np
import numpy.matlib
import matplotlib as mtpl

def show_dictionary(D):
    
	# SHOW_DICTIONARY Display a dictionary
	#
	# Input:
	#  D - input dictionary to visualize
    
	n_images = 1
	in_D_size = D.shape[0]
	each_D_size = int(in_D_size / n_images)
	all_mats = np.array([])
	for c1 in range(1, n_images + 1):
		D = D[((c1 - 1) * each_D_size + np.arange(each_D_size)).astype(int), :]
		n_atoms = D.shape[1]
		# Adding borders between the atoms
		atom_size = D.shape[0]
		block_size = round(atom_size ** 0.5)
		in_inds = np.arange(atom_size) + 1
		out_inds = np.matlib.repmat(np.array([np.arange(block_size) * (block_size + 1)]), block_size, 1) + np.matlib.repmat(np.array([1 + np.arange(block_size)]).T, 1, block_size)
		out_inds = np.reshape(out_inds.T, (out_inds.shape[0] * out_inds.shape[1], ))
		D2 = np.zeros(((block_size + 1) ** 2, n_atoms))
		D2[out_inds - 1, :] = D[in_inds - 1, :]
		remInds = np.array(list(set(1 + np.arange(D2.shape[0])) - set(out_inds)))
		D2[remInds - 1, :] = -1
		block_size = block_size + 1
		Dict = D2
		r = round(n_atoms ** 0.5)
		c = r
		final_mat = np.zeros((r * block_size, c * block_size))
		dict = np.reshape(Dict.T, (Dict.shape[0] * Dict.shape[1], 1))
		t1 = np.reshape(dict, (-1, block_size)).T # In this matrix, every blocks adjacent rows (no overlaps) are one block
		inds = np.arange(c * block_size)
		for t in range(1, r + 1):
			final_mat[(t - 1) * block_size : t * block_size, :] = t1[: , inds + (t-1) * len(inds)]
		if all_mats.size == 0:
			s = 0
		else:
			s = all_mats.shape[1]
		barrier = np.ones((4, s)) * (-1)
		barrier[1, :] = 1
		if c1 == 1:
			barrier = np.array([])
		if all_mats.size == 0:
			if barrier.size != 0:
				all_mats = np.concatenate((barrier, final_mat), axis=0)
			else:
				all_mats = final_mat
		else:
			all_mats = np.concatenate((all_mats, barrier, final_mat), axis=0)
	rng = max(abs(np.reshape(D.T, (D.shape[0] * D.shape[1], ))))
	rng = min(rng * 1.1, 1)
	mtpl.pyplot.imshow(all_mats, cmap='gray', vmin=-rng, vmax=rng)
