import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def create_image_evolution_gif(img_path, dist_path, mom_path, explore_path, truth_mom, save_path='mcmc_evolution.gif', fps=2, mom2_path=None):
	"""
	Create a GIF showing the evolution of images through MCMC iterations.
	
	Args:
		img_path: List of images from MCMC
		dist_path: List of distances
		mom_path: List of momentum tuples (x, y, z) for track 1
		explore_path: List of exploration flags
		truth_mom: Tuple of (x, y, z) truth momentum values, OR tuple of two momentum tuples for dual-track
		save_path: Path to save the GIF
		fps: Frames per second for the GIF
		mom2_path: Optional list of momentum tuples for track 2 (dual-track mode)
	"""
	
	# Detect dual-track mode
	dual_track = mom2_path is not None
	# if not dual_track and isinstance(truth_mom, tuple) and len(truth_mom) == 2 and isinstance(truth_mom[0], (np.ndarray, list, tuple)):
	# 	# truth_mom is (mom1, mom2) - dual track
	# 	dual_track = True
	# 	truth_mom1, truth_mom2 = truth_mom
	# else:
	# 	truth_mom1 = truth_mom
	# 	truth_mom2 = None
	if dual_track: 
		truth_mom1, truth_mom2 = truth_mom

	# Convert explore_path to array
	explore_array = np.array([1 if e else 0 for e in explore_path])
	
	# Create figure and axis
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	
	# Initialize with first image
	if isinstance(img_path[0], torch.Tensor):
		img_np = img_path[0].squeeze().cpu().numpy()
	else:
		img_np = np.array(img_path[0]).squeeze()
	
	im = ax.imshow(img_np, cmap='gray', aspect='equal', vmin=0, vmax=max([img.max() for img in img_path]))
	
	# Add colorbar
	cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	
	# Function to update the frame
	def update(frame):
		# Convert to numpy if tensor
		if isinstance(img_path[frame], torch.Tensor):
			img_np = img_path[frame].squeeze().cpu().numpy()
		else:
			img_np = np.array(img_path[frame]).squeeze()
		
		# Update image data
		im.set_array(img_np)
		
		# Get current momentum
		if isinstance(mom_path[frame], torch.Tensor):
			current_mom = mom_path[frame].cpu().numpy()
		elif isinstance(mom_path[frame], tuple):
			current_mom = mom_path[frame]
		else:
			current_mom = mom_path[frame]
		
		# Add border color based on status
		if frame == 0:
			color = 'blue'
			status = 'Initial'
		else:
			if explore_array[frame]:
				color = 'orange'
				status = 'EXPLORE'
			elif dist_path[frame] < dist_path[frame-1]:
				color = 'green'
				status = 'ACCEPT'
			else:
				color = 'red'
				status = 'REJECT'
		
		# Set title with momentum values
		title_text = f'Iteration {frame}: {status}\n'
		
		if dual_track:
			# Get second track momentum
			if isinstance(mom2_path[frame], torch.Tensor):
				current_mom2 = mom2_path[frame].cpu().numpy()
			elif isinstance(mom2_path[frame], tuple):
				current_mom2 = mom2_path[frame]
			else:
				current_mom2 = mom2_path[frame]
			
			title_text += f'Track 1: px={current_mom[0]:.1f}, py={current_mom[1]:.1f}, pz={current_mom[2]:.1f}\n'
			title_text += f'Track 2: px={current_mom2[0]:.1f}, py={current_mom2[1]:.1f}, pz={current_mom2[2]:.1f}\n'
			title_text += f'Distance: {dist_path[frame]:.4f}\n'
			
			# Truth momentum - handle both formats
			if truth_mom1 is not None and truth_mom2 is not None:
				# Extract scalars properly
				if isinstance(truth_mom1, (np.ndarray, torch.Tensor)):
					if isinstance(truth_mom1, torch.Tensor):
						truth_mom1_np = truth_mom1.cpu().numpy()
					else:
						truth_mom1_np = truth_mom1
					t1_x, t1_y, t1_z = float(truth_mom1_np[0]), float(truth_mom1_np[1]), float(truth_mom1_np[2])
				else:
					t1_x, t1_y, t1_z = truth_mom1[0], truth_mom1[1], truth_mom1[2]
				
				if isinstance(truth_mom2, (np.ndarray, torch.Tensor)):
					if isinstance(truth_mom2, torch.Tensor):
						truth_mom2_np = truth_mom2.cpu().numpy()
					else:
						truth_mom2_np = truth_mom2
					t2_x, t2_y, t2_z = float(truth_mom2_np[0]), float(truth_mom2_np[1]), float(truth_mom2_np[2])
				else:
					t2_x, t2_y, t2_z = truth_mom2[0], truth_mom2[1], truth_mom2[2]
				
				title_text += f'Truth T1: px={t1_x:.1f}, py={t1_y:.1f}, pz={t1_z:.1f}\n'
				title_text += f'Truth T2: px={t2_x:.1f}, py={t2_y:.1f}, pz={t2_z:.1f}'
		else:
			title_text += f'px={current_mom[0]:.1f}, py={current_mom[1]:.1f}, pz={current_mom[2]:.1f}\n'
			title_text += f'Distance: {dist_path[frame]:.4f}\n'
			
			# Truth momentum - handle both array and scalar
			if isinstance(truth_mom1, (np.ndarray, torch.Tensor)):
				if isinstance(truth_mom1, torch.Tensor):
					truth_mom1_np = truth_mom1.cpu().numpy()
				else:
					truth_mom1_np = truth_mom1
				t_x, t_y, t_z = float(truth_mom1_np[0]), float(truth_mom1_np[1]), float(truth_mom1_np[2])
			else:
				t_x, t_y, t_z = truth_mom1[0], truth_mom1[1], truth_mom1[2]
			
			title_text += f'Truth: px={t_x:.1f}, py={t_y:.1f}, pz={t_z:.1f}'
		
		ax.set_title(title_text, fontsize=11, fontweight='bold')
		
		# Color the border
		for spine in ax.spines.values():
			spine.set_edgecolor(color)
			spine.set_linewidth(4)
		
		ax.axis('off')
		
		return [im]
	
	# Create animation
	anim = FuncAnimation(fig, update, frames=len(img_path), interval=1000/fps, blit=True, repeat=True)
	
	# Save as GIF
	writer = PillowWriter(fps=fps)
	anim.save(save_path, writer=writer)
	print(f"GIF saved to {save_path}")
	
	plt.close(fig)
	return anim


def plot_images_only(img_path, dist_path, mom_path, explore_path, truth_mom, save_path='mcmc_images.png', mom2_path=None):
	"""
	Plot only the images from MCMC in a single row with square aspect ratio and gray colormap.
	
	Args:
		img_path: List of images from MCMC
		dist_path: List of distances
		mom_path: List of momentum tuples (x, y, z) for track 1
		explore_path: List of exploration flags
		truth_mom: Tuple of (x, y, z) truth momentum values, OR tuple of two momentum tuples for dual-track
		save_path: Path to save the figure
		mom2_path: Optional list of momentum tuples for track 2 (dual-track mode)
	"""
	
	# Detect dual-track mode
	dual_track = mom2_path is not None
	# if not dual_track and isinstance(truth_mom, tuple) and len(truth_mom) == 2 and isinstance(truth_mom[0], (np.ndarray, list, tuple)):
	# 	# truth_mom is (mom1, mom2) - dual track
	# 	dual_track = True
	# 	truth_mom1, truth_mom2 = truth_mom
	# else:
	# 	truth_mom1 = truth_mom
	# 	truth_mom2 = None

	# Convert explore_path to array
	explore_array = np.array([1 if e else 0 for e in explore_path])
	
	# Determine number of images to show
	n_images = len(img_path)
	
	# Create figure with images in a single row
	fig, axes = plt.subplots(1, n_images, figsize=(3*n_images, 3.5))
	
	# Make sure axes is iterable even if there's only one image
	if n_images == 1:
		axes = [axes]
	
	for i, (img, ax) in enumerate(zip(img_path, axes)):
		# Convert to numpy if tensor
		if isinstance(img, torch.Tensor):
			img_np = img.squeeze().cpu().numpy()
		else:
			img_np = np.array(img).squeeze()
		
		# Plot image with gray colormap and square aspect
		im = ax.imshow(img_np, cmap='gray', aspect='equal')
		
		# Get current momentum
		if isinstance(mom_path[i], torch.Tensor):
			current_mom = mom_path[i].cpu().numpy()
		elif isinstance(mom_path[i], tuple):
			current_mom = mom_path[i]
		else:
			current_mom = mom_path[i]
		
		# Add border color based on status
		if i == 0:
			color = 'blue'
			status = 'Initial'
		else:
			if explore_array[i]:
				color = 'orange'
				status = 'EXPLORE'
			elif dist_path[i] < dist_path[i-1]:
				color = 'green'
				status = 'ACCEPT'
			else:
				color = 'red'
				status = 'REJECT'
		
		# Set title with momentum values
		title_text = f'{status}\n'
		
		if dual_track:
			# Get second track momentum
			if isinstance(mom2_path[i], torch.Tensor):
				current_mom2 = mom2_path[i].cpu().numpy()
			elif isinstance(mom2_path[i], tuple):
				current_mom2 = mom2_path[i]
			else:
				current_mom2 = mom2_path[i]
			
			title_text += f'T1: px={current_mom[0]:.1f}, py={current_mom[1]:.1f}, pz={current_mom[2]:.1f}\n'
			title_text += f'T2: px={current_mom2[0]:.1f}, py={current_mom2[1]:.1f}, pz={current_mom2[2]:.1f}\n'
			title_text += f'Dist: {dist_path[i]:.4f}'
		else:
			title_text += f'px={current_mom[0]:.1f}, py={current_mom[1]:.1f}, pz={current_mom[2]:.1f}\n'
			title_text += f'Dist: {dist_path[i]:.4f}'
		
		ax.set_title(title_text, fontsize=9, fontweight='bold')
		
		# Color the border
		for spine in ax.spines.values():
			spine.set_edgecolor(color)
			spine.set_linewidth(3)
		
		ax.axis('off')
		
		# Add colorbar
		plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	
	if dual_track: 
		truth_mom1, truth_mom2 = truth_mom

	# Add overall title with truth values
	if dual_track and truth_mom1 is not None and truth_mom2 is not None:
		# Extract scalar values from arrays if needed
		if isinstance(truth_mom1, (np.ndarray, torch.Tensor)):
			if isinstance(truth_mom1, torch.Tensor):
				truth_mom1 = truth_mom1.cpu().numpy()
			t1_x, t1_y, t1_z = float(truth_mom1[0]), float(truth_mom1[1]), float(truth_mom1[2])
		else:
			t1_x, t1_y, t1_z = truth_mom1[0], truth_mom1[1], truth_mom1[2]
		
		if isinstance(truth_mom2, (np.ndarray, torch.Tensor)):
			if isinstance(truth_mom2, torch.Tensor):
				truth_mom2 = truth_mom2.cpu().numpy()
			t2_x, t2_y, t2_z = float(truth_mom2[0]), float(truth_mom2[1]), float(truth_mom2[2])
		else:
			t2_x, t2_y, t2_z = truth_mom2[0], truth_mom2[1], truth_mom2[2]
		
		fig.suptitle(f'MCMC Image Sequence (Dual Track)\n'
					 f'Truth T1: px={t1_x:.1f}, py={t1_y:.1f}, pz={t1_z:.1f} | '
					 f'Truth T2: px={t2_x:.1f}, py={t2_y:.1f}, pz={t2_z:.1f}', 
					 fontsize=14, fontweight='bold')
	elif truth_mom1 is not None:
		if isinstance(truth_mom1, (np.ndarray, torch.Tensor)):
			if isinstance(truth_mom1, torch.Tensor):
				truth_mom1 = truth_mom1.cpu().numpy()
			t_x, t_y, t_z = float(truth_mom1[0]), float(truth_mom1[1]), float(truth_mom1[2])
		else:
			t_x, t_y, t_z = truth_mom1[0], truth_mom1[1], truth_mom1[2]
		
		fig.suptitle(f'MCMC Image Sequence\nTruth: px={t_x:.1f}, py={t_y:.1f}, pz={t_z:.1f}', 
					 fontsize=14, fontweight='bold')
	else:
		fig.suptitle(f'MCMC Image Sequence', fontsize=14, fontweight='bold')
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	print(f"Images figure saved to {save_path}")
	
	return fig


def plot_mcmc_results_with_std(img_path, dist_path, mom_path, explore_path, std_path, truth_mom, save_dir='mcmc_plots', mom2_path=None):
	"""
	Extended visualization without images, with true momentum marked as stars.
	Saves each plot as a separate image in the specified directory.
	
	Args:
		img_path: List of images from MCMC (not used in this version)
		dist_path: List of distances
		mom_path: List of momentum tuples (x, y, z) for track 1
		explore_path: List of exploration flags
		std_path: List of momentum standard deviations (not used in this version)
		truth_mom: Tuple of (x, y, z) truth momentum values, OR tuple of two momentum tuples for dual-track
		save_dir: Directory to save individual plot images
		mom2_path: Optional list of momentum tuples for track 2 (dual-track mode)
	"""
	
	# Detect dual-track mode
	dual_track = mom2_path is not None
	# if not dual_track and isinstance(truth_mom, tuple) and len(truth_mom) == 2 and isinstance(truth_mom[0], (np.ndarray, list, tuple)):
	# 	# truth_mom is (mom1, mom2) - dual track
	# 	dual_track = True
	# 	truth_mom1, truth_mom2 = truth_mom
	# else:
	# 	truth_mom1 = truth_mom
	# 	truth_mom2 = None
	
	if dual_track: 
		truth_mom1, truth_mom2 = truth_mom

	# Extract scalar values from truth momentum if needed
	if truth_mom1 is not None:
		if isinstance(truth_mom1, (np.ndarray, torch.Tensor)):
			if isinstance(truth_mom1, torch.Tensor):
				truth_mom1 = truth_mom1.cpu().numpy()
			t1_x, t1_y, t1_z = float(truth_mom1[0]), float(truth_mom1[1]), float(truth_mom1[2])
		else:
			t1_x, t1_y, t1_z = truth_mom1[0], truth_mom1[1], truth_mom1[2]
		truth_mom1_scalar = (t1_x, t1_y, t1_z)
	else:
		truth_mom1_scalar = None
	
	if dual_track and truth_mom2 is not None:
		if isinstance(truth_mom2, (np.ndarray, torch.Tensor)):
			if isinstance(truth_mom2, torch.Tensor):
				truth_mom2 = truth_mom2.cpu().numpy()
			t2_x, t2_y, t2_z = float(truth_mom2[0]), float(truth_mom2[1]), float(truth_mom2[2])
		else:
			t2_x, t2_y, t2_z = truth_mom2[0], truth_mom2[1], truth_mom2[2]
		truth_mom2_scalar = (t2_x, t2_y, t2_z)
	else:
		truth_mom2_scalar = None
	
	# Create directory if it doesn't exist
	import os
	os.makedirs(save_dir, exist_ok=True)
	
	# Convert momentum path to arrays
	mom_array = np.array([list(m) if isinstance(m, tuple) else m.cpu().numpy() if isinstance(m, torch.Tensor) else m 
						  for m in mom_path])
	px, py, pz = mom_array[:, 0], mom_array[:, 1], mom_array[:, 2]
	
	if dual_track:
		mom2_array = np.array([list(m) if isinstance(m, tuple) else m.cpu().numpy() if isinstance(m, torch.Tensor) else m 
							   for m in mom2_path])
		px2, py2, pz2 = mom2_array[:, 0], mom2_array[:, 1], mom2_array[:, 2]
	
	# Convert explore_path to array
	explore_array = np.array([1 if e else 0 for e in explore_path])
	
	iterations = np.arange(len(dist_path))
	
	# Color code the points
	colors = ['blue'] + ['green' if not explore_array[i] and dist_path[i] < dist_path[i-1] 
						  else 'orange' if explore_array[i] 
						  else 'red' 
						  for i in range(1, len(dist_path))]
	
	from matplotlib.patches import Patch
	legend_elements = [
		Patch(facecolor='blue', edgecolor='black', label='Initial'),
		Patch(facecolor='green', edgecolor='black', label='Accepted (Better)'),
		Patch(facecolor='orange', edgecolor='black', label='Explored (Worse)'),
		Patch(facecolor='red', edgecolor='black', label='Rejected')
	]
	
	# Plot 1: Distance over iterations
	fig1, ax_dist = plt.subplots(figsize=(10, 6))
	
	ax_dist.plot(iterations, dist_path, 'k-', alpha=0.3, linewidth=1)
	ax_dist.scatter(iterations, dist_path, c=colors, s=100, zorder=5, edgecolors='black', linewidth=1.5)
	
	ax_dist.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax_dist.set_ylabel('Distance (EMD)', fontsize=12, fontweight='bold')
	
	if dual_track and truth_mom1_scalar is not None and truth_mom2_scalar is not None:
		ax_dist.set_title(f'Distance Evolution (Dual Track)\n'
						  f'Truth T1: px={truth_mom1_scalar[0]:.1f}, py={truth_mom1_scalar[1]:.1f}, pz={truth_mom1_scalar[2]:.1f} | '
						  f'Truth T2: px={truth_mom2_scalar[0]:.1f}, py={truth_mom2_scalar[1]:.1f}, pz={truth_mom2_scalar[2]:.1f}', 
						  fontsize=14, fontweight='bold')
	elif truth_mom1_scalar is not None:
		ax_dist.set_title(f'Distance Evolution\nTruth: px={truth_mom1_scalar[0]:.1f}, py={truth_mom1_scalar[1]:.1f}, pz={truth_mom1_scalar[2]:.1f}', 
						  fontsize=14, fontweight='bold')
	else:
		ax_dist.set_title('Distance Evolution', fontsize=14, fontweight='bold')
	
	ax_dist.grid(True, alpha=0.3)
	ax_dist.legend(handles=legend_elements, loc='upper right', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'distance_evolution.png'), dpi=150, bbox_inches='tight')
	print(f"Distance evolution plot saved to {save_dir}/distance_evolution.png")
	plt.close(fig1)
	
	# Plot 2: 2D Momentum trajectory (X vs Y)
	fig2, ax_mom_2d = plt.subplots(figsize=(10, 8))
	
	# Plot trajectory for track 1
	ax_mom_2d.plot(px, py, 'k-', alpha=0.3, linewidth=1, zorder=1, label='Track 1' if dual_track else None)
	
	# Plot points with colors for track 1
	for i in range(len(px)):
		color = colors[i]
		marker = 'o' if i > 0 else 's'
		size = 150 if i == 0 else 100
		ax_mom_2d.scatter(px[i], py[i], c=color, s=size, marker=marker, 
						 zorder=5, edgecolors='black', linewidth=1.5)
	
	# Add arrows to show direction for track 1
	for i in range(len(px) - 1):
		ax_mom_2d.annotate('', xy=(px[i+1], py[i+1]), xytext=(px[i], py[i]),
						   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))
	
	# Add true momentum for track 1 as a star
	if truth_mom1_scalar is not None:
		ax_mom_2d.scatter(truth_mom1_scalar[0], truth_mom1_scalar[1], c='gold', s=400, marker='*', 
						 zorder=10, edgecolors='black', linewidth=2, label='Truth T1' if dual_track else 'Truth')
	
	# If dual track, plot track 2
	if dual_track:
		# Plot trajectory for track 2
		ax_mom_2d.plot(px2, py2, 'b--', alpha=0.3, linewidth=1, zorder=1, label='Track 2')
		
		# Plot points with colors for track 2 (use different marker)
		for i in range(len(px2)):
			color = colors[i]
			marker = '^' if i > 0 else 'D'  # Different markers for track 2
			size = 150 if i == 0 else 100
			ax_mom_2d.scatter(px2[i], py2[i], c=color, s=size, marker=marker, 
							 zorder=5, edgecolors='black', linewidth=1.5, alpha=0.7)
		
		# Add arrows to show direction for track 2
		for i in range(len(px2) - 1):
			ax_mom_2d.annotate('', xy=(px2[i+1], py2[i+1]), xytext=(px2[i], py2[i]),
							   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.3, lw=1.5))
		
		# Add true momentum for track 2 as a star
		if truth_mom2_scalar is not None:
			ax_mom_2d.scatter(truth_mom2_scalar[0], truth_mom2_scalar[1], c='orange', s=400, marker='*', 
							 zorder=10, edgecolors='black', linewidth=2, label='Truth T2')
	
	ax_mom_2d.set_xlabel('Momentum X', fontsize=12, fontweight='bold')
	ax_mom_2d.set_ylabel('Momentum Y', fontsize=12, fontweight='bold')
	
	if dual_track and truth_mom1_scalar is not None and truth_mom2_scalar is not None:
		ax_mom_2d.set_title(f'2D Momentum Trajectory (X vs Y) - Dual Track\n'
							f'Truth T1: px={truth_mom1_scalar[0]:.1f}, py={truth_mom1_scalar[1]:.1f}, pz={truth_mom1_scalar[2]:.1f} | '
							f'Truth T2: px={truth_mom2_scalar[0]:.1f}, py={truth_mom2_scalar[1]:.1f}, pz={truth_mom2_scalar[2]:.1f}', 
							fontsize=14, fontweight='bold')
	elif truth_mom1_scalar is not None:
		ax_mom_2d.set_title(f'2D Momentum Trajectory (X vs Y)\nTruth: px={truth_mom1_scalar[0]:.1f}, py={truth_mom1_scalar[1]:.1f}, pz={truth_mom1_scalar[2]:.1f}', 
							fontsize=14, fontweight='bold')
	else:
		ax_mom_2d.set_title('2D Momentum Trajectory (X vs Y)', fontsize=14, fontweight='bold')
	
	ax_mom_2d.grid(True, alpha=0.3)
	ax_mom_2d.legend(loc='best', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'momentum_2d_trajectory.png'), dpi=150, bbox_inches='tight')
	print(f"2D momentum trajectory plot saved to {save_dir}/momentum_2d_trajectory.png")
	plt.close(fig2)
	
	# Plot 3: X component
	fig3, ax_mom_x = plt.subplots(figsize=(10, 6))
	
	# Track 1
	ax_mom_x.plot(iterations, px, 'k-', alpha=0.3, linewidth=1, label='Track 1' if dual_track else None)
	ax_mom_x.scatter(iterations, px, c=colors, s=100, zorder=5, edgecolors='black', linewidth=1.5)
	
	if truth_mom1_scalar is not None:
		ax_mom_x.axhline(y=truth_mom1_scalar[0], color='gold', linestyle='--', linewidth=2, 
						 label='Truth T1' if dual_track else 'Truth', zorder=1)
	
	# Track 2 if dual track
	if dual_track:
		ax_mom_x.plot(iterations, px2, 'b--', alpha=0.3, linewidth=1, label='Track 2')
		ax_mom_x.scatter(iterations, px2, c=colors, s=100, zorder=5, edgecolors='blue', 
						 linewidth=1.5, marker='^', alpha=0.7)
		
		if truth_mom2_scalar is not None:
			ax_mom_x.axhline(y=truth_mom2_scalar[0], color='orange', linestyle='--', linewidth=2, 
							 label='Truth T2', zorder=1)
	
	ax_mom_x.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax_mom_x.set_ylabel('Momentum X', fontsize=12, fontweight='bold')
	
	if dual_track and truth_mom1_scalar is not None and truth_mom2_scalar is not None:
		ax_mom_x.set_title(f'X Component Evolution\n'
						   f'Truth T1: px={truth_mom1_scalar[0]:.1f} | Truth T2: px={truth_mom2_scalar[0]:.1f}', 
						   fontsize=14, fontweight='bold')
	elif truth_mom1_scalar is not None:
		ax_mom_x.set_title(f'X Component Evolution\nTruth: px={truth_mom1_scalar[0]:.1f}', 
						   fontsize=14, fontweight='bold')
	else:
		ax_mom_x.set_title('X Component Evolution', fontsize=14, fontweight='bold')
	
	ax_mom_x.grid(True, alpha=0.3)
	ax_mom_x.legend(loc='best', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'momentum_x_component.png'), dpi=150, bbox_inches='tight')
	print(f"X component plot saved to {save_dir}/momentum_x_component.png")
	plt.close(fig3)
	
	# Plot 4: Y component
	fig4, ax_mom_y = plt.subplots(figsize=(10, 6))
	
	# Track 1
	ax_mom_y.plot(iterations, py, 'k-', alpha=0.3, linewidth=1, label='Track 1' if dual_track else None)
	ax_mom_y.scatter(iterations, py, c=colors, s=100, zorder=5, edgecolors='black', linewidth=1.5)
	
	if truth_mom1_scalar is not None:
		ax_mom_y.axhline(y=truth_mom1_scalar[1], color='gold', linestyle='--', linewidth=2, 
						 label='Truth T1' if dual_track else 'Truth', zorder=1)
	
	# Track 2 if dual track
	if dual_track:
		ax_mom_y.plot(iterations, py2, 'b--', alpha=0.3, linewidth=1, label='Track 2')
		ax_mom_y.scatter(iterations, py2, c=colors, s=100, zorder=5, edgecolors='blue', 
						 linewidth=1.5, marker='^', alpha=0.7)
		
		if truth_mom2_scalar is not None:
			ax_mom_y.axhline(y=truth_mom2_scalar[1], color='orange', linestyle='--', linewidth=2, 
							 label='Truth T2', zorder=1)
	
	ax_mom_y.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax_mom_y.set_ylabel('Momentum Y', fontsize=12, fontweight='bold')
	
	if dual_track and truth_mom1_scalar is not None and truth_mom2_scalar is not None:
		ax_mom_y.set_title(f'Y Component Evolution\n'
						   f'Truth T1: py={truth_mom1_scalar[1]:.1f} | Truth T2: py={truth_mom2_scalar[1]:.1f}', 
						   fontsize=14, fontweight='bold')
	elif truth_mom1_scalar is not None:
		ax_mom_y.set_title(f'Y Component Evolution\nTruth: py={truth_mom1_scalar[1]:.1f}', 
						   fontsize=14, fontweight='bold')
	else:
		ax_mom_y.set_title('Y Component Evolution', fontsize=14, fontweight='bold')
	
	ax_mom_y.grid(True, alpha=0.3)
	ax_mom_y.legend(loc='best', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'momentum_y_component.png'), dpi=150, bbox_inches='tight')
	print(f"Y component plot saved to {save_dir}/momentum_y_component.png")
	plt.close(fig4)
	
	# Plot 5: Z component
	fig5, ax_mom_z = plt.subplots(figsize=(10, 6))
	
	# Track 1
	ax_mom_z.plot(iterations, pz, 'k-', alpha=0.3, linewidth=1, label='Track 1' if dual_track else None)
	ax_mom_z.scatter(iterations, pz, c=colors, s=100, zorder=5, edgecolors='black', linewidth=1.5)
	
	if truth_mom1_scalar is not None:
		ax_mom_z.axhline(y=truth_mom1_scalar[2], color='gold', linestyle='--', linewidth=2, 
						 label='Truth T1' if dual_track else 'Truth', zorder=1)
	
	# Track 2 if dual track
	if dual_track:
		ax_mom_z.plot(iterations, pz2, 'b--', alpha=0.3, linewidth=1, label='Track 2')
		ax_mom_z.scatter(iterations, pz2, c=colors, s=100, zorder=5, edgecolors='blue', 
						 linewidth=1.5, marker='^', alpha=0.7)
		
		if truth_mom2_scalar is not None:
			ax_mom_z.axhline(y=truth_mom2_scalar[2], color='orange', linestyle='--', linewidth=2, 
							 label='Truth T2', zorder=1)
	
	ax_mom_z.set_xlabel('Iteration', fontsize=12, fontweight='bold')
	ax_mom_z.set_ylabel('Momentum Z', fontsize=12, fontweight='bold')
	
	if dual_track and truth_mom1_scalar is not None and truth_mom2_scalar is not None:
		ax_mom_z.set_title(f'Z Component Evolution\n'
						   f'Truth T1: pz={truth_mom1_scalar[2]:.1f} | Truth T2: pz={truth_mom2_scalar[2]:.1f}', 
						   fontsize=14, fontweight='bold')
	elif truth_mom1_scalar is not None:
		ax_mom_z.set_title(f'Z Component Evolution\nTruth: pz={truth_mom1_scalar[2]:.1f}', 
						   fontsize=14, fontweight='bold')
	else:
		ax_mom_z.set_title('Z Component Evolution', fontsize=14, fontweight='bold')
	
	ax_mom_z.grid(True, alpha=0.3)
	ax_mom_z.legend(loc='best', fontsize=10)
	
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'momentum_z_component.png'), dpi=150, bbox_inches='tight')
	print(f"Z component plot saved to {save_dir}/momentum_z_component.png")
	plt.close(fig5)
	
	print(f"\nAll plots saved to {save_dir}/")
	
	return None


# Example usage (if running standalone with test data)
if __name__ == "__main__":
	
	double = True 

	# Check if data files exist
	import os
	if os.path.exists("mcmc_outputs/img_path.pt"):
		img_path = torch.load("mcmc_outputs/img_path.pt")
		mom_path = np.load("mcmc_outputs/mom_path.npy")
		dist_path = np.load("mcmc_outputs/dist_path.npy")
		explore_path = np.load("mcmc_outputs/explore_path.npy")
		std_path = np.load("mcmc_outputs/std_path.npy")
		mom2_path = None 
		if double: 
			mom2_path = np.load("mcmc_outputs/mom2_path.npy")
	else: 
		print("No files found")
		exit() 


	# Truth momentum values (example)
	truth_mom = (314.0, -126.4, 249.1)

	colinear = np.load("/n/home11/zimani/proton64_analysis/double_momentum/angle_separated_pairs_with_emd.npy", allow_pickle=True)

	# Find a pair at ~16.1 degrees separation
	for co in colinear: 
		if np.abs(co['separation'] - 16.1) < 0.1:  
			print(f"Found pair with separation: {co['separation']}")
			double_track = co['event1']['image'] + co['event2']['image']
			mom1_true = co['event1']['momentum']
			mom2_true = co['event2']['momentum']
			truth_mom = (mom1_true, mom2_true)
			print(f"True momentum 1: {mom1_true}")
			print(f"True momentum 2: {mom2_true}")
			break 

	

	# Create GIF of image evolution
	create_image_evolution_gif(img_path, dist_path, mom_path, explore_path, truth_mom,
							   save_path='./mcmc_plots/mcmc_evolution.gif', fps=2, mom2_path=None)

	# Plot images only
	fig_images = plot_images_only(img_path, dist_path, mom_path, explore_path, truth_mom,
									save_path='./mcmc_plots/mcmc_images.png', mom2_path=mom2_path)
	
	# Plot full results (without images, without std, with truth markers)
	fig = plot_mcmc_results_with_std(img_path, dist_path, mom_path, explore_path, std_path, truth_mom,
										save_dir='./mcmc_plots/', mom2_path=mom2_path)
