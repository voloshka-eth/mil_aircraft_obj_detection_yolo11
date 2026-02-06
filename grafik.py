import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set(style="whitegrid", context="talk", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

results_path = '/path/'
output_dir = '/path/'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(results_path)

metrics = {
    'train/box_loss': 'Training Box Loss',
    'train/cls_loss': 'Training Class Loss',
    'train/dfl_loss': 'Training DFL Loss',
    'val/box_loss': 'Validation Box Loss',
    'val/cls_loss': 'Validation Class Loss',
    'val/dfl_loss': 'Validation DFL Loss',
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP@0.5',
    'metrics/mAP50-95(B)': 'mAP@0.5:0.95'
}

for metric, title in metrics.items():
    plt.figure(figsize=(10, 6))

    plt.plot(df['epoch'], df[metric],
             linewidth=2.5,
             marker='o',
             markersize=8,
             color='royalblue')

    best_epoch = df[metric].idxmax() if 'mAP' in metric or 'precision' in metric else df[metric].idxmin()
    plt.scatter(best_epoch, df[metric].iloc[best_epoch],
                color='red', s=150, zorder=5,
                label=f'Best: {df[metric].iloc[best_epoch]:.4f} (Epoch {best_epoch})')

    plt.title(f'{title} vs Epochs', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(title, fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    safe_title = title.replace('@', '_').replace(':', '_').replace('/', '_')
    plt.savefig(f'{output_dir}/{safe_title}.png', bbox_inches='tight', dpi=300)
    plt.close()

plt.figure(figsize=(10, 6))
for lr in ['lr/pg0', 'lr/pg1', 'lr/pg2']:
    plt.plot(df['epoch'], df[lr], linewidth=2.5, marker='s', markersize=6, label=lr.replace('lr/', ''))

plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Learning Rate', fontsize=14)
plt.yscale('log')
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{output_dir}/learning_rates.png', bbox_inches='tight', dpi=300)
plt.close()

