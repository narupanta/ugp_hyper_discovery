import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UQInModelDiscovery')))

from uqmodeldisc.parameterdistillation.distillation import load_normalizing_flow_parameter_distribution
from uqmodeldisc.io import ProjectDirectory
from uqmodeldisc.settings import Settings

saved_model_dir = "extraction/extracted_models/20260723T152354_nh_0.0001_0.01_1.5_0.95_5_80.0_1"
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

settings = Settings()
settings.PROJECT_DIR = Path(os.path.join(saved_model_dir, "uqmodeldisc_distilled"))
project_dir = ProjectDirectory(settings)

from distillation.distill_uqmodeldisc import PyTorchGMRModel

# We need the model instance to load the normalizing flow parameters
try:
    model = PyTorchGMRModel(num_points=1, device=device, include_log_terms=True)
    print("Loading Normalizing Flow (with log terms)...")
    distribution = load_normalizing_flow_parameter_distribution(
        model=model,
        output_subdirectory=".",
        project_directory=project_dir,
        device=device
    )
except Exception:
    print("Failed loading with 14 parameters (include_log_terms=True), retrying without log terms (12 parameters)...")
    model = PyTorchGMRModel(num_points=1, device=device, include_log_terms=False)
    distribution = load_normalizing_flow_parameter_distribution(
        model=model,
        output_subdirectory=".",
        project_directory=project_dir,
        device=device
    )

print("Drawing samples...")
# The attribute is likely just 'sample' returning a tensor, but let's check its methods.
if hasattr(distribution, 'sample'):
    samples = distribution.sample(5000)
else:
    print("No sample method found on distribution:", dir(distribution))
    sys.exit(1)

# Ensure samples are detached and numpy
if isinstance(samples, torch.Tensor):
    samples = samples.detach().cpu().numpy()

# Note: if it's a tuple or list, convert accordingly
if isinstance(samples, (list, tuple)):
    samples = torch.stack(samples, dim=0).detach().cpu().numpy()

samples_path = os.path.join(settings.PROJECT_DIR, "pytorch_flow_samples.npy")
np.save(samples_path, samples)
print(f"Saved PyTorch samples to {samples_path}")

# Now plot distributions for all parameters
parameter_names = list(model.parameter_names)

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for i, name in enumerate(parameter_names):
    ax = axes[i // 4, i % 4]
    data = samples[:, i]
    mean_val = np.mean(data)
    std_val = np.std(data)
    sns.histplot(data, ax=ax, stat='density', color='blue', alpha=0.6, bins=30)
    ax.set_title(f"{name}\nMean: {mean_val:.4f}, Std: {std_val:.4f}")
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

plt.tight_layout()
out_path = os.path.join(settings.PROJECT_DIR, "pytorch_parameter_distributions.pdf")
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
