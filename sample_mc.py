import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch 
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

class BatchedMomentumMCMC:
	def __init__(self, generator_func, reco_func, distance_func):
		"""
		Initialize Batched MCMC Image Matcher
		
		Args:
			generator_func: Function that takes batch of (x,y,z) momentum and returns batch of images
			reco_func: Function that takes batch of images and returns batch of (x,y,z) momentum
			distance_func: Function to compute distance between two images (now supports batches)
		"""
		self.generator = generator_func
		self.guesser = reco_func
		self.image_distance = distance_func
		
	def _propose_momentum_batch(self, current_momenta, step_size, batch_size):
		"""
		Propose new momentum batch using Gaussian random walk
		
		Args:
			current_momenta: Current batch of (x,y,z) momentum [(x1,y1,z1), (x2,y2,z2), ...]
			step_size: Standard deviation for proposal distribution
			batch_size: Number of proposals to generate
		
		Returns:
			Proposed new momentum batch
		"""
		proposals = []
		for i in range(batch_size):
			if i < len(current_momenta):
				# Propose around current momentum
				x, y, z = current_momenta[i]
				dx, dy, dz = np.random.normal(0, step_size, 3)
			else:
				# Generate new random proposals for additional batch slots
				base_momentum = current_momenta[0] if current_momenta else (0, 0, 0)
				x, y, z = base_momentum
				dx, dy, dz = np.random.normal(0, step_size * 2, 3)
			
			proposals.append((x + dx, y + dy, z + dz))
		
		return proposals
	
	def _acceptance_probability(self, current_distances, proposed_distances, temperature=1.0, flat_accept_prob=0.0):
		"""
		Calculate acceptance probability using Metropolis-Hastings criterion for batch
		
		Args:
			current_distances: Distances for current states
			proposed_distances: Distances for proposed states
			temperature: Temperature parameter (higher = more exploratory)
			flat_accept_prob: Flat probability to accept worse proposals (e.g., 0.1 for 10%)
		
		Returns:
			Acceptance probabilities for each proposal
		"""
		current_distances = np.array(current_distances)
		proposed_distances = np.array(proposed_distances)
		
		# Vectorized acceptance probability calculation
		improvements = proposed_distances < current_distances
		accept_probs = np.ones_like(proposed_distances, dtype=float)
		
		# For worse proposals, use either flat acceptance or exponential acceptance
		worse_mask = ~improvements
		if np.any(worse_mask):
			if flat_accept_prob > 0:
				# Use flat acceptance probability for bad guesses
				accept_probs[worse_mask] = flat_accept_prob
			else:
				# Use exponential acceptance (standard Metropolis-Hastings)
				accept_probs[worse_mask] = np.exp(
					-(proposed_distances[worse_mask] - current_distances[worse_mask]) / temperature
				)
		
		return accept_probs
	
	def run_mcmc(self, target_image, initial_momentum=None, n_iterations=1000,
				 step_size=0.1, temperature=1.0, target_distance=0.01,
				 batch_size=8, flat_accept_prob=0.0, true_momentum=None, verbose=True, plot_progress=False):
		"""
		Run Batched MCMC to find momentum that generates image close to target
		
		Args:
			target_image: Target image to match
			initial_momentum: Initial guess for momentum. If None, uses guesser
			n_iterations: Maximum number of MCMC iterations
			step_size: Step size for momentum proposals
			temperature: Temperature parameter for acceptance
			target_distance: Stop when distance falls below this threshold
			batch_size: Number of parallel proposals to evaluate per iteration
			flat_accept_prob: Flat probability to accept worse proposals (0.0 to 1.0)
			true_momentum: True momentum values (x, y, z) for plotting reference
			verbose: Print progress information
			plot_progress: Plot distance over iterations
		
		Returns:
			Dictionary with results including best momentum and convergence info
		"""
		
		# Initialize timing trackers
		time_generation = 0.0
		time_distance = 0.0
		time_reconstruction = 0.0
		
		# Initialize momentum
		if initial_momentum is None:
			t_reco_start = time.time()
			initial_momentum = self.guesser(target_image)
			time_reconstruction += time.time() - t_reco_start
			if verbose:
				print(f"Initial momentum from guesser: {initial_momentum}")
		
		# Initialize current state with single momentum
		current_momenta = [initial_momentum]
		current_images = [self.generator(initial_momentum[0], initial_momentum[1], initial_momentum[2])]
		
		# Fix tensor warning for target_image
		if isinstance(target_image, torch.Tensor):
			target_image = target_image.clone().detach().float()
		else:
			target_image = torch.tensor(target_image).float()
			
		# Fix tensor warning for initial image
		if isinstance(current_images[0], torch.Tensor):
			current_image_tensor = current_images[0].clone().detach().float()
		else:
			current_image_tensor = torch.tensor(current_images[0]).float()
			
		current_distances = [self.image_distance(target_image, current_image_tensor)]
		
		# Track best solution
		best_momentum = initial_momentum
		best_distance = current_distances[0]
		if isinstance(current_images[0], torch.Tensor):
			best_image = current_images[0].clone().detach()
		else:
			best_image = torch.tensor(current_images[0]).clone()
		
		# Track progress
		distances = [best_distance]
		momenta = [initial_momentum]
		accepted_momenta = [initial_momentum]  # Track only accepted momenta for plotting
		accepted_proposals = 0
		total_proposals = 0
		
		if verbose:
			print(f"Initial distance: {current_distances[0]:.6f}")
			print(f"Starting Batched MCMC with batch size {batch_size}...")
			if flat_accept_prob > 0:
				print(f"Using flat acceptance probability: {flat_accept_prob:.2%}")
		
		# MCMC loop
		for iteration in range(n_iterations):

			t1 = time.time() 
			# Generate batch of proposals
			proposed_momenta = self._propose_momentum_batch(
				current_momenta, step_size, batch_size
			)
			
			# Transpose to get separate lists for x, y, z coordinates
			proposed_x = [mom[0] for mom in proposed_momenta]
			proposed_y = [mom[1] for mom in proposed_momenta]
			proposed_z = [mom[2] for mom in proposed_momenta]

			# Generate images for proposed momenta (individual generation, then batch EMD)
			proposed_images = self.generator(proposed_x, proposed_y, proposed_z)
			time_generation += time.time() - t1
			if verbose and iteration % 1 == 0:
				print(f"\t Gen Time: {time.time()-t1:.3f}")

			t2 = time.time() 
			# Now calculate all EMD distances in parallel
			try:
				# Stack all proposed images into a batch tensor
				proposed_images_stacked = []
				for proposed_image in proposed_images:
					if isinstance(proposed_image, torch.Tensor):
						proposed_image_tensor = proposed_image.clone().detach().float()
					else:
						proposed_image_tensor = torch.tensor(proposed_image).float()
					
					# Ensure consistent dimensions
					if proposed_image_tensor.ndim == 2:
						proposed_image_tensor = proposed_image_tensor.unsqueeze(0)
					elif proposed_image_tensor.ndim == 3 and proposed_image_tensor.shape[0] != 1:
						proposed_image_tensor = proposed_image_tensor.squeeze()
						if proposed_image_tensor.ndim == 2:
							proposed_image_tensor = proposed_image_tensor.unsqueeze(0)
					
					proposed_images_stacked.append(proposed_image_tensor.squeeze(0) if proposed_image_tensor.shape[0] == 1 else proposed_image_tensor)
				
				# Stack into batch tensor (batch_size, height, width)
				proposed_images_batch_tensor = torch.stack(proposed_images_stacked)
				
				# Ensure target_image has correct dimensions
				target_for_batch = target_image.clone()
				if target_for_batch.ndim == 3:
					target_for_batch = target_for_batch.squeeze(0)
				
				# Create batch of target images (repeat target for each proposal)
				target_batch = target_for_batch.unsqueeze(0).repeat(len(proposed_images), 1, 1)
				
				# Calculate all distances in parallel
				proposed_distances = batch_emd_loss(target_batch, proposed_images_batch_tensor)
				
			except Exception as e:
				if verbose:
					print(f"Batch EMD calculation failed ({e}), falling back to individual EMD...")
				
				# Fallback to individual EMD calculations
				proposed_distances = []
				for proposed_image in proposed_images:
					if isinstance(proposed_image, torch.Tensor):
						proposed_image_tensor = proposed_image.clone().detach().float()
					else:
						proposed_image_tensor = torch.tensor(proposed_image).float()
					
					proposed_distance = self.image_distance(target_image, proposed_image_tensor)
					proposed_distances.append(proposed_distance)
			
			time_distance += time.time() - t2
			if verbose and iteration % 1 == 0:
				print(f"\t EMD Time: {time.time()-t2:.3f}")
			
			# Find best proposal in batch
			best_batch_idx = np.argmin(proposed_distances)
			best_batch_momentum = proposed_momenta[best_batch_idx]
			best_batch_distance = proposed_distances[best_batch_idx]
			best_batch_image = proposed_images[best_batch_idx]
			
			# Update global best if improved
			if best_batch_distance < best_distance:
				best_momentum = best_batch_momentum
				best_distance = best_batch_distance
				if isinstance(best_batch_image, torch.Tensor):
					best_image = best_batch_image.clone().detach()
				else:
					best_image = torch.tensor(best_batch_image).clone()
				
				if verbose and iteration % 1 == 0:
					print(f"New best found! Distance: {best_distance:.6f}, "
						  f"Momentum: {[f'{p:.3f}' for p in best_momentum]}")
			
			# Calculate acceptance probabilities for all proposals
			current_dist_expanded = [current_distances[0]] * batch_size
			accept_probs = self._acceptance_probability(
				current_dist_expanded, proposed_distances, temperature, flat_accept_prob
			)
			
			# Accept/reject proposals
			accepted_this_batch = 0
			for i, (accept_prob, proposed_momentum, proposed_image, proposed_distance) in enumerate(
				zip(accept_probs, proposed_momenta, proposed_images, proposed_distances)
			):
				if np.random.random() < accept_prob:
					# Accept proposal - update current state
					current_momenta = [proposed_momentum]
					current_images = [proposed_image]
					current_distances = [proposed_distance]
					accepted_momenta.append(proposed_momentum)
					accepted_this_batch += 1
					break  # Only accept one proposal per iteration to maintain single chain
			
			accepted_proposals += accepted_this_batch
			total_proposals += batch_size
			
			# Record progress (use current state, not necessarily the best)
			distances.append(current_distances[0])
			momenta.append(current_momenta[0])
			
			# Check convergence
			if best_distance < target_distance:
				if verbose:
					print(f"Converged at iteration {iteration}! "
						  f"Distance: {best_distance:.6f}")
				break
			
			# Progress reporting
			if verbose and (iteration + 1) % 1 == 0:
				acceptance_rate = accepted_proposals / total_proposals if total_proposals > 0 else 0
				print(f"Iteration {iteration + 1}: "
					  f"Current distance: {current_distances[0]:.3f}, "
					  f"Best distance: {best_distance:.3f}, "
					  f"Batch acceptance rate: {acceptance_rate:.3f}")
		
		# Final results
		final_acceptance_rate = accepted_proposals / total_proposals if total_proposals > 0 else 0
		
		# Print timing summary
		total_time = time_generation + time_distance + time_reconstruction
		if verbose:
			print(f"\n{'='*60}")
			print(f"TIMING SUMMARY:")
			print(f"{'='*60}")
			print(f"Event Generation:        {time_generation:8.3f}s  ({100*time_generation/total_time:5.1f}%)")
			print(f"Distance Calculation:    {time_distance:8.3f}s  ({100*time_distance/total_time:5.1f}%)")
			print(f"Momentum Reconstruction: {time_reconstruction:8.3f}s  ({100*time_reconstruction/total_time:5.1f}%)")
			print(f"Total Time:              {total_time:8.3f}s")
			print(f"{'='*60}")
			
			print(f"\nBatched MCMC completed!")
			print(f"Best momentum found: {best_momentum}")
			print(f"Best distance: {best_distance:.6f}")
			print(f"Final acceptance rate: {final_acceptance_rate:.3f}")
			print(f"Total proposals evaluated: {total_proposals}")
		
		# Plot progress if requested
		if plot_progress:
			plt.figure(figsize=(15, 5))
			
			# Distance convergence plot
			plt.subplot(1, 3, 1)
			plt.plot(distances, label='Current chain distance', linewidth=2)
			plt.axhline(y=best_distance, color='g', linestyle='--', 
					   label=f'Best distance: {best_distance:.4f}', linewidth=2)
			plt.axhline(y=1.0, color='r', linestyle='--', 
					   label='Target distance: 1.0', linewidth=2)
			plt.xlabel('Iteration', fontsize=11)
			plt.ylabel('Image Distance', fontsize=11)
			plt.title('Batched MCMC Convergence', fontsize=12, fontweight='bold')
			plt.legend(fontsize=10)
			plt.yscale('log')
			plt.grid(True, alpha=0.3)
			
			# X-Y momentum trajectory of ACCEPTED proposals
			plt.subplot(1, 3, 2)
			accepted_array = np.array(accepted_momenta)
			x_coords = accepted_array[:, 0]
			y_coords = accepted_array[:, 1]
			
			# Plot trajectory with color gradient showing progression
			for i in range(len(x_coords) - 1):
				alpha = 0.3 + 0.7 * (i / len(x_coords))
				plt.plot(x_coords[i:i+2], y_coords[i:i+2], 'b-', alpha=alpha, linewidth=1.5)
			
			# Mark start and end points
			plt.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
			plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End', markeredgecolor='darkred', markeredgewidth=2)
			
			# Mark best momentum point (which should be on the path)
			plt.plot(best_momentum[0], best_momentum[1], 'y*', markersize=15, 
					label='Best', markeredgecolor='orange', markeredgewidth=2, zorder=10)
			
			# Plot true momentum if provided
			if true_momentum is not None:
				plt.plot(true_momentum[0], true_momentum[1], 'kX', markersize=12, 
						label='Truth', markeredgecolor='black', markeredgewidth=2, zorder=10)
			
			plt.xlabel('x momentum', fontsize=11)
			plt.ylabel('y momentum', fontsize=11)
			plt.title('Accepted X-Y Momentum Evolution', fontsize=12, fontweight='bold')
			plt.legend(fontsize=9)
			plt.grid(True, alpha=0.3)
			
			# Z momentum evolution of ACCEPTED proposals
			plt.subplot(1, 3, 3)
			z_coords = accepted_array[:, 2]
			iterations_accepted = np.arange(len(z_coords))
			
			plt.plot(iterations_accepted, z_coords, 'o-', color='purple', alpha=0.7, markersize=5, linewidth=2, label='Accepted z')
			plt.axhline(y=best_momentum[2], color='orange', linestyle='--', linewidth=2, label=f'Best z: {best_momentum[2]:.2f}')
			
			# Plot true z momentum if provided
			if true_momentum is not None:
				plt.axhline(y=true_momentum[2], color='black', linestyle='--', linewidth=2, label=f'True z: {true_momentum[2]:.2f}')
			
			plt.xlabel('Accepted Step Number', fontsize=11)
			plt.ylabel('z momentum', fontsize=11)
			plt.title('Accepted Z Momentum Evolution', fontsize=12, fontweight='bold')
			plt.legend(fontsize=9)
			plt.grid(True, alpha=0.3)
			
			plt.tight_layout()
			plt.savefig("zprogress.png", dpi=150)
			print(f"\nPlot saved to zprogress.png")
		
		return {
			'best_momentum': best_momentum,
			'best_distance': best_distance,
			'best_image': best_image,
			'final_momentum': current_momenta[0],
			'final_distance': current_distances[0],
			'distances': distances,
			'momenta': momenta,
			'accepted_momenta': accepted_momenta,
			'acceptance_rate': final_acceptance_rate,
			'converged': best_distance < target_distance,
			'iterations_run': len(distances) - 1,
			'total_proposals': total_proposals,
			'timing': {
				'generation': time_generation,
				'distance': time_distance,
				'reconstruction': time_reconstruction,
				'total': total_time
			}
		}

background_threshold = 5e-2

### PARALLELIZED EMD DISTANCE FUNCTIONS ###

from geomloss import SamplesLoss  

def weights_and_positions_batch(batch_matrices):
	"""
	Extract weights and positions for a batch of matrices
	
	Args:
		batch_matrices: Tensor of shape (batch_size, height, width)
	
	Returns:
		List of (weights, positions) tuples for each matrix in the batch
	"""
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
	"""
	Compute EMD loss for batches of images in parallel
	
	Args:
		target_batch: Tensor of shape (batch_size, height, width) - target images
		proposed_batch: Tensor of shape (batch_size, height, width) - proposed images
	
	Returns:
		List of EMD distances for each pair
	"""
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

### REST OF YOUR ORIGINAL CODE ###

def run_batched_example():
	"""Run a complete batched example with parallelized EMD"""
	print("Running Batched MCMC Image Matching Example with Parallel EMD")
	print("=" * 60)
	
	# Create target image with known momentum
	x, y, z = 314.0, -126.4, 249.1  # sample 1 
	target_image = ldm_generator(x, y, z)

	print(f"True momentum: {(x, y, z)}")
	
	# Initialize Batched MCMC matcher with original single EMD function for fallback
	matcher = BatchedMomentumMCMC(ldm_generator, reco_model, emd_loss)
	
	# Run Batched MCMC with 10% flat acceptance probability for bad guesses
	results = matcher.run_mcmc(
		target_image=target_image,
		n_iterations=10,
		step_size=3, # sigma of dx 
		temperature=0.1,
		target_distance=0.01,
		batch_size=16,  # Evaluate 16 proposals per iteration
		flat_accept_prob=0.10,  # 10% flat acceptance for worse proposals
		true_momentum=(x, y, z),  # Pass true momentum for plotting
		verbose=True,
		plot_progress=True
	)
	
	print(f"\nResults:")
	print(f"True momentum:  {(x, y, z)}")
	print(f"Found momentum: {results['best_momentum']}")
	error = np.linalg.norm(np.array(results['best_momentum']) - np.array([x, y, z]))
	print(f"Error: {error:.4f}")
	print(f"Speedup factor: ~{results['total_proposals'] / results['iterations_run']:.1f}x proposals per iteration")
	print(f"Total accepted proposals: {len(results['accepted_momenta'])}")
	
	return results

### Reco Model ## 
## Hack to fix imports 
import torch
sys.path.append('/n/home11/zimani/reco_model/')
from ResNet.ResNet import ResNet50 # reco momentum model 

# Load model and weights 
reco_model_checkpoint = '/n/home11/zimani/reco_model/checkpoints/ResNet50_edep_v1/ResNet50_epoch38.pt'
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
		checkpoint_path="/n/home11/zimani/latent-diffusion/edep_protons64_ldm/runs/checkpoints/epoch=000040.ckpt",
		save_plot=False,
		verbose=False)
	batch[batch < background_threshold] = 0.0
	batch = torch.tensor(batch)
	if batch.ndim == 2: 
		batch = batch.unsqueeze(0)
	return torch.tensor(batch) 


if __name__ == "__main__":
	# Run the batched example with parallelized EMD
	results = run_batched_example()