import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import base64
import io
import numpy as np

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='black', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def mask_to_colored(mask_slice):
    colored = np.zeros((*mask_slice.shape, 3), dtype=np.float32)
    colored[mask_slice == 1] = [1, 0, 0]   # Red   — NCR
    colored[mask_slice == 2] = [0, 1, 0]   # Green — ED
    colored[mask_slice == 3] = [0, 0, 1]   # Blue  — ET
    return colored

def create_flair_image(flair_slice):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(flair_slice.T, cmap='gray', origin='lower')
    ax.axis('off')
    return fig_to_base64(fig)

def create_predicted_image(pred_slice):
    pred_colored = mask_to_colored(pred_slice)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(pred_colored.transpose(1, 0, 2), origin='lower')
    ax.axis('off')
    return fig_to_base64(fig)

def create_overlay_image(flair_slice, pred_slice):
    pred_colored = mask_to_colored(pred_slice)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(flair_slice.T, cmap='gray', origin='lower')
    ax.imshow(pred_colored.transpose(1, 0, 2), alpha=0.6, origin='lower')
    ax.axis('off')
    return fig_to_base64(fig)

def create_gt_image(gt_slice):
    gt_colored = mask_to_colored(gt_slice)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('black')
    ax.imshow(gt_colored.transpose(1, 0, 2), origin='lower')
    ax.axis('off')
    return fig_to_base64(fig)
