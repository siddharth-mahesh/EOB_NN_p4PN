import re
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_logs(file_path):
    epochs = []
    train_losses = []
    val_losses = []
    
    # RelAbsComp components (Train and Val)
    rel_abs_comp_train = []
    rel_abs_comp_val = []
    
    # Regex to match the epoch summary lines
    epoch_pattern = re.compile(
        r"\[HybridEOB\] Epoch (?P<epoch>\d+), Loss: (?P<loss>[\d.e+-]+), Val Loss: (?P<val_loss>[\d.e+-]+)"
    )
    
    # Pattern for RelAbsComp
    # Example: RelAbsComp(train,val): [T1 T2 T3 T4] [V1 V2 V3 V4]
    rel_abs_pattern = re.compile(
        r"RelAbsComp\(train,val\): \[(?P<train_vals>.*?)\] \[(?P<val_vals>.*?)\]"
    )

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    with open(file_path, 'r') as f:
        current_epoch_data = None
        for line in f:
            e_match = epoch_pattern.search(line)
            if e_match:
                current_epoch_data = {
                    'epoch': int(e_match.group('epoch')),
                    'loss': float(e_match.group('loss')),
                    'val_loss': float(e_match.group('val_loss')),
                }
                continue
            
            if current_epoch_data is not None:
                r_match = rel_abs_pattern.search(line)
                if r_match:
                    t_vals_str = r_match.group('train_vals').strip()
                    v_vals_str = r_match.group('val_vals').strip()
                    
                    t_vals = [float(x) for x in t_vals_str.split()]
                    v_vals = [float(x) for x in v_vals_str.split()]
                    
                    epochs.append(current_epoch_data['epoch'])
                    train_losses.append(current_epoch_data['loss'])
                    val_losses.append(current_epoch_data['val_loss'])
                    rel_abs_comp_train.append(t_vals)
                    rel_abs_comp_val.append(v_vals)
                    
                    current_epoch_data = None

    return {
        'epoch': epochs,
        'loss': train_losses,
        'val_loss': val_losses,
        'rel_abs_comp_train': np.array(rel_abs_comp_train) if rel_abs_comp_train else None,
        'rel_abs_comp_val': np.array(rel_abs_comp_val) if rel_abs_comp_val else None
    }

def plot_logs(data, output_path='training_evolution.png'):
    if not data or not data['epoch']:
        print("No data found to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Losses
    axes[0].plot(data['epoch'], data['loss'], label='Train Loss', alpha=0.8, marker='.')
    axes[0].plot(data['epoch'], data['val_loss'], label='Val Loss', alpha=0.8, marker='.')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Loss (Log Scale)')
    axes[0].set_title('Training and Validation Loss Evolution')
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.5)

    # Plot RelAbsComp components (Train and Val)
    if data['rel_abs_comp_train'] is not None and data['rel_abs_comp_val'] is not None:
        num_components = data['rel_abs_comp_val'].shape[1]
        labels = [r'$\dot{r}$', r'$\dot{\phi}$', r'$\dot{p}_r$', r'$\dot{p}_\phi$']
        colors = plt.cm.tab10(np.linspace(0, 1, num_components))
        
        for i in range(num_components):
            label = labels[i] if i < len(labels) else f'Comp {i+1}'
            color = colors[i]
            # Plot Train as dashed
            axes[1].plot(data['epoch'], data['rel_abs_comp_train'][:, i], 
                         linestyle='--', color=color, alpha=0.5, label=f'{label}')
            # Plot Val as solid
            axes[1].plot(data['epoch'], data['rel_abs_comp_val'][:, i], 
                         linestyle='-', color=color, alpha=0.8)
    
    axes[1].plot(np.nan, np.nan, linestyle='--', color='black', alpha=0.5, label='Train')
    axes[1].plot(np.nan, np.nan, linestyle='-', color='black', alpha=0.8, label='Val')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Relative Error (Log Scale)')
    axes[1].set_title('Component-wise relative: Train (dashed) vs Val (solid)')
    # Put legend outside to avoid cluttering if too many lines
    axes[1].legend(loc='upper right',framealpha=0.5)
    axes[1].grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    log_file = '/home/sidmahesh/EOB_NN_p4PN/most_recent_training_program.txt'
    output_path = 'visualizations/recent_training_evolution.png'
    data = parse_logs(log_file)
    if data:
        plot_logs(data, output_path)


