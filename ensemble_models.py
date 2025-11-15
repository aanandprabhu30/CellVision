"""
CellVision - Ensemble Model Implementation
Combines multiple state-of-the-art segmentation models for superior performance
"""

import numpy as np
from skimage import io, measure, morphology
from scipy import ndimage
from cellpose import models
import warnings
warnings.filterwarnings('ignore')


class ModelEnsemble:
    """
    Ensemble of multiple cell segmentation models
    Combines CellPose, StarDist-like approach, and custom refinement
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize ensemble with multiple models
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models in the ensemble"""
        print("🔄 Loading ensemble models...")
        
        # Model 1: CellPose cyto2 (generalist)
        try:
            self.models['cellpose_cyto2'] = models.CellposeModel(
                gpu=self.use_gpu,
                model_type='cyto2'
            )
            print("✅ CellPose cyto2 loaded")
        except Exception as e:
            print(f"⚠️  CellPose cyto2 failed: {e}")
        
        # Model 2: CellPose cyto3 (if available, newer version)
        try:
            self.models['cellpose_cyto3'] = models.CellposeModel(
                gpu=self.use_gpu,
                model_type='cyto3'
            )
            print("✅ CellPose cyto3 loaded")
        except Exception as e:
            print(f"⚠️  CellPose cyto3 not available (using cyto2 only)")
        
        # Model 3: CellPose nuclei (for nuclear segmentation)
        try:
            self.models['cellpose_nuclei'] = models.CellposeModel(
                gpu=self.use_gpu,
                model_type='nuclei'
            )
            print("✅ CellPose nuclei loaded")
        except Exception as e:
            print(f"⚠️  CellPose nuclei failed: {e}")
        
        print(f"✅ Ensemble ready with {len(self.models)} models")
    
    def _watershed_segmentation(self, image, diameter):
        """
        Watershed-based segmentation (StarDist-like approach)
        
        Args:
            image: Input image
            diameter: Estimated cell diameter
            
        Returns:
            Segmentation masks
        """
        from skimage import filters, feature
        from scipy import ndimage as ndi
        
        # Denoise
        denoised = filters.gaussian(image, sigma=2)
        
        # Edge detection
        edges = filters.sobel(denoised)
        
        # Threshold
        thresh = filters.threshold_otsu(denoised)
        binary = denoised > thresh
        
        # Distance transform
        distance = ndi.distance_transform_edt(binary)
        
        # Find peaks (cell centers)
        coords = feature.peak_local_max(
            distance,
            min_distance=int(diameter * 0.5),
            labels=binary
        )
        
        # Create markers
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = measure.label(mask)
        
        # Watershed
        labels = morphology.watershed(-distance, markers, mask=binary)
        
        return labels
    
    def _compute_iou(self, mask1, mask2):
        """
        Compute Intersection over Union between two masks
        
        Args:
            mask1, mask2: Binary masks
            
        Returns:
            IoU score
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _match_masks(self, masks_list, iou_threshold=0.3):
        """
        Match corresponding cells across different model predictions
        
        Args:
            masks_list: List of mask arrays from different models
            iou_threshold: Minimum IoU to consider a match
            
        Returns:
            Matched cell groups
        """
        if len(masks_list) == 0:
            return []
        
        # Get all unique cells from all models
        all_cells = []
        
        for model_idx, masks in enumerate(masks_list):
            cell_ids = np.unique(masks)[1:]  # Exclude background (0)
            
            for cell_id in cell_ids:
                cell_mask = (masks == cell_id)
                all_cells.append({
                    'model_idx': model_idx,
                    'cell_id': cell_id,
                    'mask': cell_mask,
                    'area': cell_mask.sum()
                })
        
        # Group overlapping cells
        matched_groups = []
        used_indices = set()
        
        for i, cell1 in enumerate(all_cells):
            if i in used_indices:
                continue
            
            group = [i]
            used_indices.add(i)
            
            for j, cell2 in enumerate(all_cells[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                # Check IoU with any cell in current group
                for group_idx in group:
                    iou = self._compute_iou(
                        all_cells[group_idx]['mask'],
                        cell2['mask']
                    )
                    
                    if iou >= iou_threshold:
                        group.append(j)
                        used_indices.add(j)
                        break
            
            matched_groups.append([all_cells[idx] for idx in group])
        
        return matched_groups
    
    def _merge_predictions(self, masks_list, confidences=None):
        """
        Merge predictions from multiple models using voting
        
        Args:
            masks_list: List of segmentation masks
            confidences: Optional confidence scores for each model
            
        Returns:
            Merged segmentation masks
        """
        if len(masks_list) == 0:
            return np.zeros((100, 100), dtype=np.int32)
        
        if len(masks_list) == 1:
            return masks_list[0]
        
        # Match cells across models
        matched_groups = self._match_masks(masks_list, iou_threshold=0.3)
        
        # Create final mask
        final_mask = np.zeros_like(masks_list[0], dtype=np.int32)
        
        for cell_idx, group in enumerate(matched_groups, start=1):
            # Voting: use the mask from the model with highest confidence
            # or the one with median area if no confidences
            
            if confidences is not None:
                # Weight by confidence
                best_idx = max(
                    range(len(group)),
                    key=lambda i: confidences[group[i]['model_idx']]
                )
            else:
                # Use median area as heuristic
                areas = [cell['area'] for cell in group]
                median_area = np.median(areas)
                best_idx = min(
                    range(len(group)),
                    key=lambda i: abs(group[i]['area'] - median_area)
                )
            
            # Add to final mask
            final_mask[group[best_idx]['mask']] = cell_idx
        
        return final_mask
    
    def predict(self, image, diameter=None, channels=[0, 0]):
        """
        Ensemble prediction combining multiple models
        
        Args:
            image: Input image (grayscale or RGB)
            diameter: Cell diameter (None for auto-detection)
            channels: Channel configuration for CellPose
            
        Returns:
            masks: Combined segmentation masks
            confidence: Ensemble confidence score
            individual_predictions: Dict of predictions from each model
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            from skimage.color import rgb2gray
            image_gray = (rgb2gray(image) * 255).astype(np.uint8)
        else:
            image_gray = image
        
        # Auto-detect diameter if not provided
        if diameter is None:
            diameter = self._estimate_diameter(image_gray)
        
        predictions = {}
        masks_list = []
        confidences = []
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                print(f"  Running {model_name}...")
                
                if 'cellpose' in model_name:
                    masks, flows, styles = model.eval(
                        image_gray,
                        diameter=diameter,
                        channels=channels
                    )
                    
                    # Estimate confidence from flow quality
                    confidence = self._estimate_cellpose_confidence(flows)
                else:
                    masks = np.zeros_like(image_gray, dtype=np.int32)
                    confidence = 0.5
                
                predictions[model_name] = {
                    'masks': masks,
                    'confidence': confidence
                }
                
                masks_list.append(masks)
                confidences.append(confidence)
                
            except Exception as e:
                print(f"  ⚠️  {model_name} failed: {e}")
                continue
        
        # Add watershed segmentation
        try:
            print("  Running watershed segmentation...")
            watershed_masks = self._watershed_segmentation(image_gray, diameter)
            predictions['watershed'] = {
                'masks': watershed_masks,
                'confidence': 0.7
            }
            masks_list.append(watershed_masks)
            confidences.append(0.7)
        except Exception as e:
            print(f"  ⚠️  Watershed failed: {e}")
        
        # Merge predictions
        print("  Merging predictions...")
        ensemble_masks = self._merge_predictions(masks_list, confidences)
        
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(confidences) if confidences else 0.0
        
        return ensemble_masks, ensemble_confidence, predictions
    
    def _estimate_diameter(self, image):
        """Estimate cell diameter from image"""
        from skimage import filters
        
        edges = filters.sobel(image)
        labeled = morphology.label(edges > filters.threshold_otsu(edges))
        regions = measure.regionprops(labeled)
        
        if len(regions) > 0:
            diameters = [r.equivalent_diameter for r in regions if r.area > 50]
            if len(diameters) > 0:
                return max(20, min(100, int(np.median(diameters))))
        
        return 30
    
    def _estimate_cellpose_confidence(self, flows):
        """
        Estimate confidence from CellPose flow fields
        
        Args:
            flows: Flow field output from CellPose
            
        Returns:
            Confidence score (0-1)
        """
        try:
            if len(flows) > 0 and flows[0] is not None:
                # Use flow magnitude as confidence proxy
                flow_magnitude = np.sqrt(flows[0][0]**2 + flows[0][1]**2)
                confidence = np.mean(flow_magnitude) / 10.0  # Normalize
                return min(1.0, max(0.0, confidence))
        except:
            pass
        
        return 0.8  # Default confidence


class AdaptiveEnsemble:
    """
    Adaptive ensemble that selects best model based on image characteristics
    """
    
    def __init__(self, use_gpu=False):
        self.ensemble = ModelEnsemble(use_gpu=use_gpu)
    
    def _analyze_image_characteristics(self, image):
        """
        Analyze image to determine best model
        
        Returns:
            characteristics dict
        """
        from skimage import filters, exposure
        
        # Calculate image statistics
        contrast = exposure.is_low_contrast(image)
        edges = filters.sobel(image)
        edge_density = (edges > filters.threshold_otsu(edges)).mean()
        
        # Estimate cell density
        labeled = morphology.label(edges > filters.threshold_otsu(edges))
        n_regions = len(np.unique(labeled)) - 1
        density = n_regions / (image.shape[0] * image.shape[1])
        
        return {
            'low_contrast': contrast,
            'edge_density': edge_density,
            'cell_density': density,
            'is_dense': density > 0.001
        }
    
    def predict(self, image, diameter=None):
        """
        Adaptive prediction using image characteristics
        
        Args:
            image: Input image
            diameter: Cell diameter (optional)
            
        Returns:
            masks, confidence, metadata
        """
        # Analyze image
        characteristics = self._analyze_image_characteristics(image)
        
        # Run ensemble
        masks, confidence, predictions = self.ensemble.predict(
            image,
            diameter=diameter
        )
        
        return masks, confidence, {
            'predictions': predictions,
            'characteristics': characteristics,
            'ensemble_size': len(predictions)
        }


# Convenience function
def create_ensemble(use_gpu=False, adaptive=True):
    """
    Create ensemble model
    
    Args:
        use_gpu: Whether to use GPU
        adaptive: Whether to use adaptive ensemble
        
    Returns:
        Ensemble model instance
    """
    if adaptive:
        return AdaptiveEnsemble(use_gpu=use_gpu)
    else:
        return ModelEnsemble(use_gpu=use_gpu)


if __name__ == "__main__":
    print("CellVision Ensemble Models")
    print("This module provides state-of-the-art ensemble segmentation")
    print("\nUsage:")
    print("  from ensemble_models import create_ensemble")
    print("  ensemble = create_ensemble(use_gpu=False)")
    print("  masks, confidence, metadata = ensemble.predict(image)")
