import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import time

from geomloss import SamplesLoss


def weights_and_positions(matrix):
	"""Extract weights and positions from a 2D matrix."""
	positions = torch.nonzero(matrix, as_tuple=False).float()
	if len(positions) == 0:
		# Handle empty matrix
		return torch.tensor([]), torch.tensor([])
	weights = matrix[positions[:, 0].long(), positions[:, 1].long()]
	return weights, positions


def emd_loss(a, b):
	"""Calculate EMD loss between two 2D images."""
	# Ensure 2D
	if a.ndim > 2:
		raise ValueError("Input 'a' must be 2D")
	if b.ndim > 2:
		raise ValueError("Input 'b' must be 2D")
	
	a_w, a_p = weights_and_positions(a)
	b_w, b_p = weights_and_positions(b)
	
	# Handle empty cases
	if len(a_w) == 0 or len(b_w) == 0:
		return 0.0
	
	# Normalize weights for balanced EMD
	a_w = a_w / a_w.sum()
	b_w = b_w / b_w.sum()
	
	EMD = SamplesLoss("sinkhorn", p=1, blur=0.01)
	
	# Prevent single sample at same location
	if len(a_p) == 1 and len(b_p) == 1 and torch.equal(a_p, b_p):
		distance = 0
	else:
		distance = EMD(a_w, a_p, b_w, b_p).item()
	
	return distance


def batch_emd_loss(target, proposed_batch):
	"""Calculate average EMD loss between target and batch of proposed images."""
	distances = []
	for i in range(proposed_batch.shape[0]):
		dist = emd_loss(target, proposed_batch[i])
		distances.append(dist)
	return np.mean(distances)


if __name__ == "__main__":
	batch = np.load("batch_0.npy")  # shape (128, 64, 64)
	target_image = batch[0]
	proposed_images = batch[1:64]  # 3 images
	
	background_threshold = 5e-2
	target_image[target_image < background_threshold] = 0
	proposed_images[proposed_images < background_threshold] = 0

	# Convert to torch tensors
	target_tensor = torch.tensor(target_image, dtype=torch.float32)
	proposed_tensor = torch.tensor(proposed_images, dtype=torch.float32)
	
	print(f"Target shape: {target_tensor.shape}")
	print(f"Proposed batch shape: {proposed_tensor.shape}")
	print()
	
	# Method 1: Full images (original method)
	print("=" * 50)
	print("Method 1: Processing full images")
	print("=" * 50)
	start_time = time.time()
	proposed_distance = batch_emd_loss(target_tensor, proposed_tensor)
	elapsed_time = time.time() - start_time
	print(f"Average Distance: {proposed_distance:.6f}")
	print(f"Time taken: {elapsed_time:.6f} seconds")
	print()
	
	# Method 2: Extract nonzero regions first (pre-processing)
	print("=" * 50)
	print("Method 2: Pre-extracting nonzero info")
	print("=" * 50)
	start_time = time.time()
	
	# Get nonzero indices for target
	target_nz_idx = np.nonzero(target_image)
	num_target_nz = len(target_nz_idx[0])
	
	# Get nonzero info for proposed images
	proposed_nz_counts = []
	for i in range(proposed_images.shape[0]):
		proposed_nz_idx = np.nonzero(proposed_images[i])
		proposed_nz_counts.append(len(proposed_nz_idx[0]))
	
	print(f"Target nonzero elements: {num_target_nz}")
	print(f"Proposed nonzero elements: {proposed_nz_counts}")
	
	# The actual EMD computation still uses the full tensors
	# (weights_and_positions already extracts nonzero internally)
	proposed_distance_v2 = batch_emd_loss(target_tensor, proposed_tensor)
	elapsed_time_v2 = time.time() - start_time
	print(f"Average Distance: {proposed_distance_v2:.6f}")
	print(f"Time taken: {elapsed_time_v2:.6f} seconds")
	print()
	
	# Method 3: Show per-image timing
	print("=" * 50)
	print("Method 3: Per-image timing breakdown")
	print("=" * 50)
	start_time = time.time()
	for i in range(proposed_tensor.shape[0]):
		img_start = time.time()
		dist = emd_loss(target_tensor, proposed_tensor[i])
		img_time = time.time() - img_start
		print(f"Image {i+1}: distance={dist:.6f}, time={img_time:.6f}s")
	elapsed_time_v3 = time.time() - start_time
	print(f"Total time: {elapsed_time_v3:.6f} seconds")
	print()
	
	# Summary
	print("=" * 50)
	print("SUMMARY")
	print("=" * 50)
	print(f"Method 1 (batch processing):  {elapsed_time:.6f}s")
	print(f"Method 2 (with nonzero info): {elapsed_time_v2:.6f}s")
	print(f"Method 3 (per-image timing):  {elapsed_time_v3:.6f}s")
	# print(f"\nAll methods produce same result: {np.allclose(proposed_distance, proposed_distance_v2)}")