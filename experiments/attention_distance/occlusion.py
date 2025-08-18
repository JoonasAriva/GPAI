import numpy as np
import torch


def occlusion_sensitivity_batch(model, image, scan_end, target_class=None, patch_size=15, stride=8, baseline=0.0,
                                batch_size=64,
                                device="cuda", ):
    """
    Batched Occlusion Sensitivity for a single image.

    Args:
        model: PyTorch model
        image: Tensor (C, H, W) in range [0,1] or normalized
        target_class: int, class index to evaluate. If None, use model's top prediction.
        patch_size: size of square occlusion patch
        stride: step size for sliding the patch
        baseline: value to fill the patch (e.g., 0.0 for black)
        batch_size: number of occluded images per forward pass
        device: "cuda" or "cpu"

    Returns:
        saliency_map: (H, W) numpy array
    """
    model.eval().to(device)
    image = image.unsqueeze(0).to(device)  # Add batch dim
    C, H, W = image.shape[1:]

    with torch.no_grad():
        out = model(image, scan_end=scan_end, cam=True)
        output = out['attention_weights']
        # if target_class is None:
        #    target_class = output.argmax(dim=1).item()
        base_score = output.item()

    saliency_map_att = np.zeros((H, W), dtype=np.float32)
    saliency_map_cls = np.zeros((H, W), dtype=np.float32)
    occluded_images = []
    coords = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_occ = image.clone()

            img_occ[:, :, y:y + patch_size, x:x + patch_size] = baseline
            occluded_images.append(img_occ)
            coords.append((y, x))

            # Process in batches
            if len(occluded_images) == batch_size:

                batch_tensor = torch.cat(occluded_images, dim=0).to(device)
                with torch.no_grad():
                    outs = model(batch_tensor, scan_end=5000, cam=True)
                    outputs_att = outs['attention_weights'].flatten()
                    outputs_cls = outs['scores'].flatten()

                    scores_att = outputs_att.cpu().numpy()
                    scores_cls = outputs_cls.cpu().numpy()

                for (cy, cx), score_att, score_cls in zip(coords, scores_att, scores_cls):
                    diff_att = base_score - score_att
                    diff_cls = base_score - score_cls
                    saliency_map_att[cy:cy + patch_size, cx:cx + patch_size] += diff_att
                    saliency_map_cls[cy:cy + patch_size, cx:cx + patch_size] += diff_cls
                occluded_images.clear()
                coords.clear()

    # Process leftover batch
    if occluded_images:

        batch_tensor = torch.cat(occluded_images, dim=0).to(device)
        with torch.no_grad():
            outs = model(batch_tensor, scan_end=5000, cam=True)
            outputs_att = outs['attention_weights'].flatten()
            outputs_cls = outs['scores'].flatten()
            # print("outputs:", outputs.shape)
            scores_att = outputs_att.cpu().numpy()
            scores_cls = outputs_cls.cpu().numpy()
        for (cy, cx), score_att, score_cls in zip(coords, scores_att, scores_cls):
            diff_att = base_score - score_att
            diff_cls = base_score - score_cls
            saliency_map_att[cy:cy + patch_size, cx:cx + patch_size] += diff_att
            saliency_map_cls[cy:cy + patch_size, cx:cx + patch_size] += diff_cls
    saliency_map_att[saliency_map_att < 0] = 0
    saliency_map_cls[saliency_map_cls < 0] = 0
    # Normalize to [0,1]
    saliency_map_att -= saliency_map_att.min()
    saliency_map_att /= saliency_map_att.max() + 1e-8

    saliency_map_cls -= saliency_map_cls.min()
    saliency_map_cls /= saliency_map_cls.max() + 1e-8

    return saliency_map_att, saliency_map_cls
