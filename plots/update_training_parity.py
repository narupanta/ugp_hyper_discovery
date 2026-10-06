import os
import json
import yaml
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from core.material_models import get_material
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.features import IsotropicFeatureExtractor
from core.utils import farthest_point_sampling_with_fixed_point
from plots.training import plot_training_r2


def update_seed(exp_dir, seed):
    save_path = os.path.join(exp_dir, str(seed), 'extracted')
    cfg_path = os.path.join(save_path, 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_path = f"/home/mmdiscovery/shared/dataset/preprocessed/syn_f/isihara_0.0005_0.05_1.0_0.5_block_{seed}.npz"
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("/home/mmdiscovery/shared", cfg.get("dataset_path", ""))
    data = np.load(dataset_path)

    dev_params = cfg['dev_params']
    vol_params = cfg['vol_params']
    true_mat = get_material('isihara', dev_params=dev_params, vol_params=vol_params)

    best_params_dict = np.load(os.path.join(save_path, 'best_params.npy'), allow_pickle=True).item()
    valid_keys = set(GPRawParams._fields)
    filtered_params = {k: v for k, v in best_params_dict.items() if k in valid_keys}
    raw_params = GPRawParams(**filtered_params)

    extractor = IsotropicFeatureExtractor()
    F_train_full = jnp.array(data['F_3d'])

    train_steps = cfg['train_load_steps_indices']
    val_steps = cfg['val_load_steps_indices']
    test_steps = cfg.get('test_load_steps_indices', None)

    dev_flat = F_train_full[jnp.array(train_steps)].reshape(-1, 3, 3)
    dev_extracted, vol_extracted = jax.vmap(extractor.extract)(dev_flat)
    n_ip = cfg['n_ip']
    dev_z = farthest_point_sampling_with_fixed_point(dev_extracted, n_ip, jnp.array([3.0, 3.0]))
    vol_z = farthest_point_sampling_with_fixed_point(vol_extracted, n_ip, jnp.array([1.0]))
    I_z = jnp.concatenate([dev_z, vol_z], axis=-1)

    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    model = SparseHyperelasticityGP(
        raw_params=raw_params,
        I_z=I_z,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        sampling_mode='pathwise',
        beta=cfg['beta'],
        L=cfg['num_rff'],
        feature_extractor=extractor,
        covariance_mode=cfg['covariance_mode'],
        normalize_ell=cfg['normalize_ell'],
        u_var_anchor=float(cfg['u_var_anchor']),
        kzz_jitter=float(cfg['kzz_jitter']),
        constraint_lengthscale=cfg['constraint_lengthscale']
    )

    r2_res = plot_training_r2(
        model, true_mat, F_train_full, save_path,
        train_steps=train_steps,
        val_steps=val_steps,
        test_steps=test_steps,
        n_pws_samples=256
    )

    print(f"Seed {seed}:")
    print(f"  Train: {r2_res.train_metrics['steps']}")
    print(f"  Val:   {r2_res.val_metrics['steps']}")
    print(f"  Test:  {r2_res.test_metrics['steps']}")

    metrics_path = os.path.join(save_path, 'extraction_metrics.json')
    with open(metrics_path, 'r') as f:
        em = json.load(f)

    em['r2'] = r2_res.train_metrics.get('r2')
    em['rmse'] = r2_res.train_metrics.get('rmse')
    em['ec'] = r2_res.train_metrics.get('ec')
    em['r2_train'] = r2_res.train_metrics.get('r2')
    em['rmse_train'] = r2_res.train_metrics.get('rmse')
    em['ec_train'] = r2_res.train_metrics.get('ec')
    em['r2_energy_val'] = r2_res.val_metrics.get('r2')
    em['rmse_energy_val'] = r2_res.val_metrics.get('rmse')
    em['ec_energy_val'] = r2_res.val_metrics.get('ec')
    em['r2_test'] = r2_res.test_metrics.get('r2')
    em['rmse_test'] = r2_res.test_metrics.get('rmse')
    em['ec_test'] = r2_res.test_metrics.get('ec')
    em['train_steps'] = r2_res.train_metrics.get('steps', train_steps)
    em['val_steps'] = r2_res.val_metrics.get('steps', val_steps)
    em['test_steps'] = r2_res.test_metrics.get('steps', [])

    with open(metrics_path, 'w') as f:
        json.dump(em, f, indent=4)


if __name__ == '__main__':
    exp_dir = '/home/mmdiscovery/shared/results/20261002T223217_isihara_0.0005_0.05_1.0_0.5_5_1.0_isotropic_block'
    for entry in sorted(os.listdir(exp_dir)):
        if entry.isdigit() and os.path.exists(os.path.join(exp_dir, entry, 'extracted', 'best_params.npy')):
            print(f"=== Updating Seed {entry} ===")
            update_seed(exp_dir, int(entry))

