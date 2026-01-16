import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch 
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def run_mcmc(generator, dist_func, reco_func=None, true_img=None, true_mom=None, 
			 batch_size=8, num_iters=10, momentum_std=5.0, momentum_std_min=0.5, 
			 std_decay='linear', explore_prob=0.1, min_distance=0.0001): 
	"""
	Run MCMC with batch generation at each iteration.
	
	Args:
		generator: Function that generates images from momentum (accepts lists of x, y, z values)
		dist_func: Distance function (should handle batches)
		reco_func: Optional reconstruction function to get initial momentum
		true_img: True/target image to match
		true_mom: True momentum values (if known)
		batch_size: Number of proposals to generate per iteration
		num_iters: Number of MCMC iterations
		momentum_std: Initial standard deviation for momentum perturbations
		momentum_std_min: Minimum standard deviation (final value)
		std_decay: Decay schedule for momentum_std ('linear', 'exponential', or 'cosine')
		explore_prob: Probability of accepting worse solution
		min_distance: Minimum distance threshold to stop early (default: 0.0001)
	"""
	
	# Handle true image and momentum
	if true_img is None: 
		true_img = generator(*true_mom)

	if true_mom is None and reco_func is not None:
		true_mom = reco_func(true_img)
	elif true_mom is None: 
		true_mom = [0, 0, 0]  # Unknown true momentum 
	
	# Ensure tensors
	true_img = torch.tensor(true_img) if not isinstance(true_img, torch.Tensor) else true_img
	true_mom = torch.tensor(true_mom) if not isinstance(true_mom, torch.Tensor) else true_mom
	
	# Grid search initial guesses for momentum
	values = np.linspace(-500, 500, batch_size)
	grid_momenta = [(x, y, z) for x, y, z in zip(values, values, values)]
	
	# Generate images for all grid points
	x_list = [m[0] for m in grid_momenta]
	y_list = [m[1] for m in grid_momenta]
	z_list = [m[2] for m in grid_momenta]
	
	grid_batch_raw = generator(x_list, y_list, z_list)
	
	# Convert to proper format
	if isinstance(grid_batch_raw, torch.Tensor):
		if grid_batch_raw.ndim == 3:
			grid_images = [grid_batch_raw[i] for i in range(batch_size)]
		elif grid_batch_raw.ndim == 4:
			grid_images = [grid_batch_raw[i] for i in range(batch_size)]
	else:
		grid_images = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img 
					   for img in grid_batch_raw]
	
	# Stack for batch distance computation
	grid_batch = torch.stack(grid_images)
	
	# Expand target image to match batch size
	if true_img.ndim == 4:
		target_batch = true_img.expand(batch_size, -1, -1, -1)
	elif true_img.ndim == 3:
		target_batch = true_img.unsqueeze(0).expand(batch_size, -1, -1, -1)
	else:
		target_batch = true_img.unsqueeze(0).expand(batch_size, -1, -1)
	
	# Make sure dimensions match
	if grid_batch.ndim != target_batch.ndim:
		if grid_batch.ndim == 3 and target_batch.ndim == 4:
			grid_batch = grid_batch.unsqueeze(1)
		elif grid_batch.ndim == 4 and target_batch.ndim == 3:
			target_batch = target_batch.unsqueeze(1)

	# Compute distances for grid search
	grid_distances = dist_func(target_batch, grid_batch)
	
	# Find best initial guess
	best_idx = np.argmin(grid_distances)
	init_mom = grid_momenta[best_idx]
	init_img = grid_images[best_idx]
	init_dist = grid_distances[best_idx]
	
	print(f"Grid search complete. Best initial momentum: {init_mom}, Distance: {init_dist:.6f}")

	img_path = [init_img]
	dist_path = [init_dist]
	mom_path = [init_mom]
	explore_path = [False]
	std_path = []  # Track momentum std over iterations

	for iteration in range(num_iters):
		# Check if minimum distance reached
		if dist_path[-1] <= min_distance:
			print(f"\nMinimum distance threshold ({min_distance}) reached at iteration {iteration}!")
			print(f"Final distance: {dist_path[-1]:.6f}")
			break
		
		# Anneal momentum_std over iterations
		progress = iteration / max(num_iters - 1, 1)  # 0 to 1
		
		if std_decay == 'linear':
			current_std = momentum_std + (momentum_std_min - momentum_std) * progress
		elif std_decay == 'exponential':
			current_std = momentum_std * (momentum_std_min / momentum_std) ** progress
		elif std_decay == 'cosine':
			current_std = momentum_std_min + 0.5 * (momentum_std - momentum_std_min) * (1 + np.cos(np.pi * progress))
		else:
			current_std = momentum_std  # No decay
		
		std_path.append(current_std)
		
		# Generate batch of proposed momenta by adding Gaussian noise
		current_mom = np.array(mom_path[-1])
		noise = np.random.normal(0, current_std, size=(batch_size, 3))
		proposed_momenta_array = current_mom + noise
		
		# Separate into x, y, z lists for batch generation
		x_list = proposed_momenta_array[:, 0].tolist()
		y_list = proposed_momenta_array[:, 1].tolist()
		z_list = proposed_momenta_array[:, 2].tolist()
		
		# Generate entire batch at once
		proposed_batch_raw = generator(x_list, y_list, z_list)
		
		# Convert to proper format (list of tensors)
		if isinstance(proposed_batch_raw, torch.Tensor):
			if proposed_batch_raw.ndim == 3:
				# Single batch output: (batch_size, height, width)
				proposed_images = [proposed_batch_raw[i] for i in range(batch_size)]
			elif proposed_batch_raw.ndim == 4:
				# Batch with channel: (batch_size, channels, height, width)
				proposed_images = [proposed_batch_raw[i] for i in range(batch_size)]
		else:
			# If it's already a list or array
			proposed_images = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img 
							   for img in proposed_batch_raw]
		
		# Stack for batch distance computation
		proposed_batch = torch.stack(proposed_images)
		
		# Store momenta as tuples for consistency
		proposed_momenta = [tuple(proposed_momenta_array[i]) for i in range(batch_size)]
		
		# Expand target image to match batch size - handle different dimensions properly
		if true_img.ndim == 4:
			# Already has batch dimension
			target_batch = true_img.expand(batch_size, -1, -1, -1)
		elif true_img.ndim == 3:
			# Has channel dimension
			target_batch = true_img.unsqueeze(0).expand(batch_size, -1, -1, -1)
		else:
			# Just height x width
			target_batch = true_img.unsqueeze(0).expand(batch_size, -1, -1)
		
		# Make sure proposed_batch has same number of dimensions as target_batch
		if proposed_batch.ndim != target_batch.ndim:
			if proposed_batch.ndim == 3 and target_batch.ndim == 4:
				proposed_batch = proposed_batch.unsqueeze(1)
			elif proposed_batch.ndim == 4 and target_batch.ndim == 3:
				target_batch = target_batch.unsqueeze(1)
		
		# Compute distances for entire batch
		proposed_distances = dist_func(target_batch, proposed_batch)
		
		# Find best proposal
		best_idx = np.argmin(proposed_distances)
		best_distance = proposed_distances[best_idx]
		best_momentum = proposed_momenta[best_idx]
		best_image = proposed_images[best_idx]
		
		# Decide whether to explore (accept worse solution)
		explore = np.random.rand() < explore_prob
		
		# Accept best proposal or explore randomly
		if best_distance < dist_path[-1]:
			# Accept best proposal (improvement)
			img_path.append(torch.tensor(best_image) if not isinstance(best_image, torch.Tensor) else best_image)
			dist_path.append(best_distance)
			mom_path.append(best_momentum)
			explore_path.append(False)
			status = "ACCEPTED (better)"
		elif explore:
			# Explore: randomly pick a proposal (not necessarily the best)
			explore_idx = np.random.randint(batch_size)
			img_path.append(torch.tensor(proposed_images[explore_idx]) if not isinstance(proposed_images[explore_idx], torch.Tensor) else proposed_images[explore_idx])
			dist_path.append(proposed_distances[explore_idx])
			mom_path.append(proposed_momenta[explore_idx])
			explore_path.append(True)
			status = "EXPLORED (worse)"
		else:
			# Reject: keep current state
			status = "REJECTED"
		
		print(f"Iteration {iteration+1}/{num_iters}: Std: {current_std:.3f}, "
			  f"Best Dist: {best_distance:.6f}, Current Dist: {dist_path[-1]:.6f}, "
			  f"Status: {status}, Batch Mean: {np.mean(proposed_distances):.6f}")
	
	# Final summary
	print(f"\nMCMC Complete!")
	print(f"Initial distance: {dist_path[0]:.6f}")
	print(f"Final distance: {dist_path[-1]:.6f}")
	print(f"Improvement: {dist_path[0] - dist_path[-1]:.6f}")
	print(f"Initial momentum: {mom_path[0]}")
	print(f"Final momentum: {mom_path[-1]}")
	if true_mom is not None:
		print(f"True momentum: {tuple(true_mom.cpu().numpy() if isinstance(true_mom, torch.Tensor) else true_mom)}")
	
	return img_path, dist_path, mom_path, explore_path, std_path

from geomloss import SamplesLoss  

def weights_and_positions_batch(batch_matrices):
	batch_data = []
	for i in range(batch_matrices.shape[0]):
		matrix = batch_matrices[i]
		positions = torch.nonzero(matrix).float()
		if len(positions) > 0:
			weights = matrix[positions[:, 0].long(), positions[:, 1].long()]
		else:
			# Handle empty matrix case
			positions = torch.tensor([[0.0, 0.0]])
			weights = torch.tensor([1e-8])  # Small epsilon to avoid division by zero
		batch_data.append((weights, positions))
	
	return batch_data

def batch_emd_loss(target_batch, proposed_batch):
	# Ensure proper dimensions
	if target_batch.ndim == 4:
		target_batch = target_batch.squeeze(1)  # Remove channel dimension if present
	if proposed_batch.ndim == 4:
		proposed_batch = proposed_batch.squeeze(1)  # Remove channel dimension if present
	
	batch_size = target_batch.shape[0]
	distances = []
	
	# Extract weights and positions for all targets and proposals
	target_data = weights_and_positions_batch(target_batch)
	proposed_data = weights_and_positions_batch(proposed_batch)
	
	# Initialize EMD loss function (reuse for efficiency)
	EMD = SamplesLoss("sinkhorn", p=1, blur=0.01)
	
	# Process all pairs in the batch
	for i in range(batch_size):
		try:
			a_w, a_p = target_data[i]
			b_w, b_p = proposed_data[i]
			
			# Normalize weights for balanced EMD
			a_w = a_w / a_w.sum()
			b_w = b_w / b_w.sum()
			
			# Prevent single sample at same location
			if len(a_p) == 1 and len(b_p) == 1 and torch.equal(a_p, b_p):
				distance = 0.0
			else:
				distance = EMD(a_w, a_p, b_w, b_p).item()
		
		except Exception as e:
			# Fallback for problematic cases
			print(f"EMD calculation failed for batch item {i}: {e}")
			distance = float('inf')
		
		distances.append(distance)
	
	return distances


def batch_emd_loss_vectorized(target_batch, proposed_batch):
	"""
	Fully vectorized EMD loss computation (more advanced version)
	This version tries to compute multiple EMDs simultaneously when possible
	
	Args:
		target_batch: Tensor of shape (batch_size, height, width)
		proposed_batch: Tensor of shape (batch_size, height, width)
	
	Returns:
		List of EMD distances
	"""
	batch_size = target_batch.shape[0]
	distances = []
	
	# Try to group similar sparsity patterns for vectorized computation
	EMD = SamplesLoss("sinkhorn", p=1, blur=0.01)
	
	# Process in mini-batches of compatible sparsity patterns
	for i in range(batch_size):
		target = target_batch[i]
		proposed = proposed_batch[i]
		
		# Squeeze dimensions if needed
		if target.ndim == 3:
			target = target.squeeze()
		if proposed.ndim == 3:
			proposed = proposed.squeeze()
		
		# Extract positions and weights
		target_positions = torch.nonzero(target).float()
		target_weights = target[target_positions[:, 0].long(), target_positions[:, 1].long()]
		
		proposed_positions = torch.nonzero(proposed).float()
		proposed_weights = proposed[proposed_positions[:, 0].long(), proposed_positions[:, 1].long()]
		
		# Handle edge cases
		if len(target_positions) == 0:
			target_positions = torch.tensor([[0.0, 0.0]])
			target_weights = torch.tensor([1e-8])
		if len(proposed_positions) == 0:
			proposed_positions = torch.tensor([[0.0, 0.0]])
			proposed_weights = torch.tensor([1e-8])
		
		# Normalize weights
		target_weights = target_weights / target_weights.sum()
		proposed_weights = proposed_weights / proposed_weights.sum()
		
		# Compute EMD
		if len(target_positions) == 1 and len(proposed_positions) == 1 and torch.equal(target_positions, proposed_positions):
			distance = 0.0
		else:
			try:
				distance = EMD(target_weights, target_positions, proposed_weights, proposed_positions).item()
			except Exception as e:
				print(f"Vectorized EMD failed for item {i}: {e}")
				distance = float('inf')
		
		distances.append(distance)
	
	return distances

def batch_l2_loss(target_batch, proposed_batch):
    """
    Compute L2 (Euclidean) distance between target and proposed batches.
    This measures the square root of sum of squared differences.
    """
    # Ensure proper dimensions
    if target_batch.ndim == 4:
        target_batch = target_batch.squeeze(1)  # Remove channel dimension if present
    if proposed_batch.ndim == 4:
        proposed_batch = proposed_batch.squeeze(1)  # Remove channel dimension if present
    
    batch_size = target_batch.shape[0]
    distances = []
    
    # Process all pairs in the batch
    for i in range(batch_size):
        target = target_batch[i]
        proposed = proposed_batch[i]
        
        # Compute L2 distance (Euclidean norm)
        distance = torch.norm(target - proposed, p=2).item()
        distances.append(distance)
    
    return distances


def batch_mse_loss(target_batch, proposed_batch):
    """
    Compute Mean Squared Error between target and proposed batches.
    This measures the average of squared differences across all elements.
    """
    # Ensure proper dimensions
    if target_batch.ndim == 4:
        target_batch = target_batch.squeeze(1)  # Remove channel dimension if present
    if proposed_batch.ndim == 4:
        proposed_batch = proposed_batch.squeeze(1)  # Remove channel dimension if present
    
    batch_size = target_batch.shape[0]
    distances = []
    
    # Process all pairs in the batch
    for i in range(batch_size):
        target = target_batch[i]
        proposed = proposed_batch[i]
        
        # Compute MSE
        distance = torch.mean((target - proposed) ** 2).item()
        distances.append(distance)
    
    return distances

# Original single-image EMD function (kept for compatibility)
def weights_and_positions(matrix): 
	positions = torch.nonzero(matrix).float()
	weights = matrix[positions[:, 0].long(), positions[:, 1].long()]
	return weights, positions 

def emd_loss(a, b): 
	if a.ndim == 3: 
		a = torch.squeeze(a) 
	if b.ndim == 3:
		b = torch.squeeze(b)

	a_w, a_p = weights_and_positions(a)
	b_w, b_p = weights_and_positions(b)

	## Normalize weights for balanced EMD
	a_w = a_w / a_w.sum()
	b_w = b_w / b_w.sum()

	EMD = SamplesLoss("sinkhorn", p=1, blur=0.01)

	## Prevent single sample at same location 
	if len(a_p) == 1 and len(b_p) == 1 and torch.equal(a_p, b_p):
		distance = 0 
	else: 
		distance = EMD(a_w, a_p, b_w, b_p).item()
	return distance 

background_threshold = 5e-2

### Reco Model ## 
## Hack to fix imports 
import torch
sys.path.append('/n/home11/zimani/reco_model/')
from ResNet.ResNet import ResNet50 # reco momentum model 

# Load model and weights 
reco_model_checkpoint = '/n/home11/zimani/reco_model/checkpoints/ResNet50_edep_v4/ResNet50_epoch100.pt'
model = ResNet50(num_classes=3, channels=1, norm='batch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(torch.load(reco_model_checkpoint, weights_only=True)['model_state_dict'])
model.eval() 

def reco_model(batch): 
	# model_input = torch.tensor(batch).unsqueeze(1).to(device)  # Add batch and channel dimensions
	if batch.ndim == 2:
		model_input = torch.tensor(batch).unsqueeze(0).unsqueeze(0).to(device) 
	if batch.ndim == 3:
		model_input = torch.tensor(batch).unsqueeze(0).to(device)
	else: 
		model_input = torch.tensor(batch).to(device)
	with torch.no_grad():
		pred = model(model_input)
	reco_mom = pred.squeeze().cpu().numpy() * 500 
	return reco_mom

## LDM Generator ### 
sys.path.append("/n/home11/zimani/latent-diffusion") 
from gen_cLDM import generate_conditioned_samples

def ldm_generator(x, y, z): 
	batch = generate_conditioned_samples(
		px=x, py=y, pz=z,
		n_samples=1,
		n_iters=1, 
		config_path="/n/home11/zimani/latent-diffusion/configs/latent-diffusion/protons64-ldm-kl.yaml",
		checkpoint_path="/n/home11/zimani/latent-diffusion/edep_protons64_v2_ldm/runs/checkpoints/epoch=000075.ckpt",
		save_plot=False,
		verbose=False)
	batch[batch < background_threshold] = 0.0
	batch = torch.tensor(batch)
	if batch.ndim == 2: 
		batch = batch.unsqueeze(0)
	return torch.tensor(batch) 


if __name__ == "__main__":

	x, y, z = 314.0, -126.4, 249.1  # sample 1 truth momentum
	sample1 = np.load("sample1.npy")

	colinear = np.load("/n/home11/zimani/proton64_analysis/double_momentum/angle_separated_pairs_with_emd.npy", allow_pickle=True)

	print(colinear.shape)

	# events_data.append({
	# 		'image': img,
	# 		'momentum': mom,
	# 		'angle': rad_angle,
	# 		'length': length,
	# 		'width': width,
	# 		'batch_num': batch_num,
	# 		'idx': idx
	# })

	# angle_pairs.append({
	# 	'event1': event1,
	# 	'event2': event2,
	# 	'separation': angle_diff_deg,
	# 	'target_separation': target_sep,
	# 	'midpoint_angle': mid_angle_deg
	# })

	for co in colinear: 
		if np.abs(co['separation'] - 16.1) < 0.1:  
			print(co['separation'])

			double_track = co['event1']['image'] + co['event2']['image']
			break 

	# plt.imshow(double_track, cmap='gray')
	# plt.savefig("tmp.png")
	# plt.clf() 

	# exit() 

	# sample1 = ldm_generator(x,y,z)

	# print("Reco")
	
	# print(reco_model(sample11)) # batch 

	# print(reco_model(sample1)) # ldm = [ 324.81577  -125.445786  409.66034 ]

	# img_path = torch.load("mcmc_outputs/img_path.pt")

	# mcmc_img = np.array(img_path[-1].unsqueeze(0))

	# print(reco_model(mcmc_img))

	# for img in img_path:

	# 	print(reco_model(np.array(img.unsqueeze(0))))

	# exit()



	# Now use batch_emd_loss instead of emd_loss for batch processing
	img_path, dist_path, mom_path, explore_path, std_path = run_mcmc(
		generator=ldm_generator, 
		dist_func=batch_emd_loss,  # Changed to batch version
		# reco_model,
		reco_func=None, 
		# true_img=sample1,  # Changed from init_img to true_img
		# true_mom=[x, y, z],  # Changed from init_mom to true_mom
		true_img=double_track,
		true_mom=None,
		batch_size=32,  # Process 16 proposals per iteration
		num_iters=16,
		momentum_std=50.0,  # Initial std
		momentum_std_min=5.0,  # Final std
		std_decay='cosine',  # Options: 'linear', 'exponential', 'cosine', or None
		explore_prob=0.1,
		min_distance=0.01  # Stop early if distance reaches this threshold
	)
	
	print("FINAL RECO MOM")
	print(reco_model(np.array(img_path[-1].unsqueeze(0))))

	if True: 
		# Save outputs for use in plotting script 
		torch.save(torch.stack(img_path), "mcmc_outputs/img_path.pt")
		np.save("mcmc_outputs/mom_path.npy", np.array(mom_path))
		np.save("mcmc_outputs/dist_path.npy", np.array(dist_path))
		np.save("mcmc_outputs/explore_path.npy", np.array(explore_path))
		np.save("mcmc_outputs/std_path.npy", np.array(std_path))

	print("DONE MCMC")

	truth_mom = (x,y,z)

	# Plot results
	from plot_mcmc import plot_images_only, plot_mcmc_results_with_std, create_image_evolution_gif
	
	# Plot images in separate figure with gray colormap
	create_image_evolution_gif(img_path, dist_path, mom_path, explore_path, truth_mom,
								save_path='mcmc_evolution.gif', fps=2)

	# Plot images only
	fig_images = plot_images_only(img_path, dist_path, mom_path, explore_path, truth_mom,
									save_path='mcmc_images.png')
	
	# Plot full results (without images, without std, with truth markers)
	fig = plot_mcmc_results_with_std(img_path, dist_path, mom_path, explore_path, std_path, truth_mom,
										save_dir='./mcmc_plots')
	
	print("DONE PLOTTING")