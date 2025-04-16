import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import torch

def plot_batch(images, labels, class_names, max_images=16):
    """
    Plot a batch of images with their corresponding labels.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): Batch of labels in YOLO format
        class_names (list): List of class names
        max_images (int): Maximum number of images to plot
    """
    n = min(images.shape[0], max_images)
    fig, axes = plt.subplots(n//4 + (n%4>0), 4, figsize=(15, 3*n//4))
    axes = axes.flatten()
    
    for idx in range(n):
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        # Draw bounding boxes
        for label in labels[idx]:
            class_id, x, y, w, h = label
            class_name = class_names[int(class_id)]
            
            x1 = int((x - w/2) * img.shape[1])
            y1 = int((y - h/2) * img.shape[0])
            x2 = int((x + w/2) * img.shape[1])
            y2 = int((y + h/2) * img.shape[0])
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics):
    """
    Plot training metrics over epochs.
    
    Args:
        metrics (dict): Dictionary containing training metrics
            Expected keys: 'train_loss', 'val_loss', 'map', 'precision', 'recall'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(metrics['train_loss'], label='Training Loss')
    ax1.plot(metrics['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot metrics
    ax2.plot(metrics['map'], label='mAP')
    ax2.plot(metrics['precision'], label='Precision')
    ax2.plot(metrics['recall'], label='Recall')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Metrics')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def visualize_predictions(image, predictions, class_names, conf_threshold=0.5):
    """
    Visualize model predictions on an image.
    
    Args:
        image (numpy.ndarray): Input image
        predictions (torch.Tensor): Model predictions (x1, y1, x2, y2, conf, class_id)
        class_names (list): List of class names
        conf_threshold (float): Confidence threshold for displaying predictions
    """
    img_copy = image.copy()
    
    if len(predictions):
        for pred in predictions:
            if pred[4] >= conf_threshold:  # Check confidence threshold
                x1, y1, x2, y2, conf, class_id = pred
                class_name = class_names[int(class_id)]
                
                # Draw bounding box
                cv2.rectangle(img_copy, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Add label
                label = f'{class_name} {conf:.2f}'
                cv2.putText(img_copy, label, 
                           (int(x1), int(y1-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
    
    return img_copy 