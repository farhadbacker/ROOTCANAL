import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class YOLOGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.model
        self.gradients = None
        self.activations = None
        
        # If no layer is specified, try to find the last convolutional layer in the last block
        # For YOLOv8, this is typically part of the SPPF or the last C2f block in the backbone/neck
        if target_layer is None:
            # This is a heuristic to find a good layer. 
            # In YOLOv8, layer -2 is usually the detection head, so we look a bit before that.
            # We'll try to hook into the last C2f module (usually layer 9 or similar in backbone, 
            # or layers in the neck).
            # Let's try to target the very last module before the Detect head.
            # The detect head is usually model.model[-1]
            try:
                # Target the last bottleneck of the backbone or neck
                # Accessing internal modules list
                # Inspecting model to find a suitable layer
                layers = list(self.model.modules())
                # We want a Conv2d layer that is deep in the network
                target_layer = layers[-2] # A fallback
                
                # Iterate backwards to find a Conv2d or C2f
                for layer in reversed(layers):
                    if isinstance(layer, torch.nn.Conv2d):
                        target_layer = layer
                        break
            except:
                pass
        
        self.target_layer = target_layer
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def __call__(self, img_tensor, target_category_index=None):
        # img_tensor: preprocessed image (1, 3, H, W)
        
        # 1. Forward pass
        # We need to run the model such that we can call backward.
        # Ultralytics model() call wraps a lot of things. We might need to run model.model() directly.
        # However, model.model(img) returns detailed output.
        
        # Ensure gradients are enabled
        self.model.eval() # Keep in eval mode for correct BN/Dropout behavior
        self.model.zero_grad()
        
        # KEY FIX: Input tensor must require gradients to build the backend graph
        img_tensor.requires_grad_(True)
        
        # Run forward
        with torch.set_grad_enabled(True):
            # Hack: Force parameters to allow gradients to avoid "Inference tensors" error
            # This is often needed when using YOLOv8 in inference mode setups
            original_requires_grad = {}
            for name, param in self.model.named_parameters():
                original_requires_grad[name] = param.requires_grad
                param.requires_grad = True
            
            # FIX: Clone anchors/strides to avoid "Inference tensors" error
            # This happens because previous inference runs created these tensors in inference_mode
            for m in self.model.modules():
                if hasattr(m, 'anchors') and isinstance(m.anchors, torch.Tensor):
                    m.anchors = m.anchors.clone()
                if hasattr(m, 'strides') and isinstance(m.strides, torch.Tensor):
                    m.strides = m.strides.clone()

            try:
                preds = self.model(img_tensor)
            finally:
                # Restore original state (optional but good practice)
                for name, param in self.model.named_parameters():
                    param.requires_grad = original_requires_grad.get(name, False)
        
        # preds is usually a list. The actual detection output is often [0]
        # In YOLOv8, output is [Batch, 4 + Num_Classes, Anchors]
        
        output = preds[0] 
        
        # 2. Select Target
        # output shape: (1, 84, 8400) for YOLOv8n (4 box + 80 classes = 84)
        # For our custom model: (1, 4 + num_classes, anchors)
        
        # We need to find the box with the highest confidence for the target category
        # or just the max confidence across all if None.
        
        # Transpose to (1, Anchors, Classes+Box)
        output = output.transpose(1, 2) # (1, 8400, 84)
        
        # Extract class scores (ignoring 4 box coords)
        # Box coords are 0-3, scores start at 4
        class_scores = output[..., 4:]
        
        if target_category_index is None:
            # Find max score across all classes and anchors
            target_score = torch.max(class_scores)
        else:
             # Find max score for specific class
            target_score = torch.max(class_scores[..., target_category_index])
            
        # 3. Backward pass
        target_score.backward()
        
        # 4. Generate Heatmap
        gradients = self.gradients # [1, C, H, W]
        activations = self.activations # [1, C, H, W]
        
        # Global Average Pooling of gradients (weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # [1, C, 1, 1]
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # [1, 1, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0]

def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB heatmap
    heatmap_uint8 = (255 * heatmap).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    # img is expected to be BGR (OpenCV standard)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
        
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed
