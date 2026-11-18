"""Loss monitoring plot of a distillation run (loss history npz written by uqmodeldisc)."""
import os


def plot_loss_monitoring(npz_path, title_suffix, output_path, window_size=100, last_n_iterations=None):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        if not os.path.exists(npz_path):
            print(f"Loss file {npz_path} not found, skipping loss monitoring plot.")
            return
        data = np.load(npz_path)
        w1_loss = data['w1']
        lip_loss = data['lipschitz']
        iterations = np.arange(1, len(w1_loss) + 1)
        
        has_grad_penalty = 'grad_penalty' in data

        df = pd.DataFrame({'w1': w1_loss, 'lip': lip_loss})
        w1_ma = df['w1'].rolling(window=window_size, min_periods=1).mean().values
        w1_var = df['w1'].rolling(window=window_size, min_periods=1).var().fillna(0).values
        lip_ma = df['lip'].rolling(window=window_size, min_periods=1).mean().values
        lip_var = df['lip'].rolling(window=window_size, min_periods=1).var().fillna(0).values

        if has_grad_penalty:
            grad_penalty = data['grad_penalty']
            df['gp'] = grad_penalty
            gp_ma = df['gp'].rolling(window=window_size, min_periods=1).mean().values
            gp_var = df['gp'].rolling(window=window_size, min_periods=1).var().fillna(0).values

        if last_n_iterations is not None:
            if len(iterations) <= last_n_iterations:
                return
            w1_loss = w1_loss[-last_n_iterations:]
            lip_loss = lip_loss[-last_n_iterations:]
            iterations = iterations[-last_n_iterations:]
            w1_ma = w1_ma[-last_n_iterations:]
            w1_var = w1_var[-last_n_iterations:]
            lip_ma = lip_ma[-last_n_iterations:]
            lip_var = lip_var[-last_n_iterations:]
            if has_grad_penalty:
                grad_penalty = grad_penalty[-last_n_iterations:]
                gp_ma = gp_ma[-last_n_iterations:]
                gp_var = gp_var[-last_n_iterations:]

        nrows = 3 if has_grad_penalty else 2
        figsize = (12, 15) if has_grad_penalty else (12, 10)
        
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        if nrows == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
        # Plot 1: Wasserstein Distance
        ax1.plot(iterations, w1_loss, color='#bdc3c7', alpha=0.4, label='W1 Distance (Raw)')
        ax1.plot(iterations, w1_ma, color='#2980b9', linewidth=2, label=f'W1 Distance (MA, window={window_size})')
        ax1_right = ax1.twinx()
        ax1_right.plot(iterations, w1_var, color='#e74c3c', linestyle='--', alpha=0.75, label=f'W1 Variance (window={window_size})')
        ax1.set_ylabel('Wasserstein Distance (W1)', fontsize=12, fontweight='bold', color='#2980b9')
        ax1_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#e74c3c')
        ax1.set_title(f"Wasserstein Loss & Oscillation Monitoring {title_suffix}", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        # Plot 2: Lipschitz Critic Function Loss
        ax2.plot(iterations, lip_loss, color='#d2b4de', alpha=0.4, label='Lipschitz Function Loss (Raw)')
        ax2.plot(iterations, lip_ma, color='#8e44ad', linewidth=2, label=f'Lipschitz Loss (MA, window={window_size})')
        ax2_right = ax2.twinx()
        ax2_right.plot(iterations, lip_var, color='#e67e22', linestyle='--', alpha=0.75, label=f'Lipschitz Variance (window={window_size})')
        if not has_grad_penalty:
            ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Lipschitz Function Loss', fontsize=12, fontweight='bold', color='#8e44ad')
        ax2_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#e67e22')
        ax2.set_title(f"Lipschitz Critic Loss & Oscillation Monitoring {title_suffix}", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)

        lines3, labels3 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right', fontsize=10)

        if has_grad_penalty:
            # Plot 3: Gradient Penalty
            ax3.plot(iterations, grad_penalty, color='#f5cba7', alpha=0.4, label='Gradient Penalty (Raw)')
            ax3.plot(iterations, gp_ma, color='#d35400', linewidth=2, label=f'Gradient Penalty (MA, window={window_size})')
            ax3_right = ax3.twinx()
            ax3_right.plot(iterations, gp_var, color='#c0392b', linestyle='--', alpha=0.75, label=f'Penalty Variance (window={window_size})')
            ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Gradient Penalty', fontsize=12, fontweight='bold', color='#d35400')
            ax3_right.set_ylabel('Variance / Oscillation', fontsize=11, fontweight='bold', color='#c0392b')
            ax3.set_title(f"Gradient Penalty Monitoring {title_suffix}", fontsize=14, fontweight='bold')
            ax3.grid(True, linestyle=':', alpha=0.6)
            
            lines5, labels5 = ax3.get_legend_handles_labels()
            lines6, labels6 = ax3_right.get_legend_handles_labels()
            ax3.legend(lines5 + lines6, labels5 + labels6, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved loss monitoring plot to {output_path}")
    except Exception as e:
        print(f"Error generating loss monitoring plot: {e}")
