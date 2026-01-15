import timm
import torch
import os
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    @torch.no_grad()
    def predict_image(img_path, topk=5):
        # Load and transform
        img = Image.open(img_path).convert("RGB")
        tensor = transforms(img).unsqueeze(0).to(device)  # shape: [1, C, H, W]

        # Forward pass
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        # Top-k predictions
        topk_probs, topk_indices = probs.topk(topk, dim=1)
        topk_probs = topk_probs[0].cpu().tolist()
        topk_indices = topk_indices[0].cpu().tolist()

        results = []
        for p, idx in zip(topk_probs, topk_indices):
            label = class_names[idx] if "class_names" in globals() and idx < len(class_names) else str(idx)
            results.append((label, p))
        return results

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    input_size = None
    model_name = "mobilenetv3_large_100"
    model = timm.create_model(model_name, pretrained=False, num_classes=4)  # or pretrained=False then load your weights
    model.eval()
    model.to(device)

    tmp_model = timm.create_model(model_name, pretrained=True)
    cfg = timm.data.resolve_data_config(tmp_model.pretrained_cfg) if input_size is None else {"input_size": input_size}
    transforms = timm.data.create_transform(**cfg, is_training=False)

    ckpt_path = "best_scene_model.pth"  # change this
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict["model_state"])
    print("Loaded fine-tuned weights from", ckpt_path)

    class_names = ["bridge", "others", "railway", "ship"]  # change this if needed

    grandparent_dir = Path(__file__).resolve().parents[2]
    data_dir = os.path.join(grandparent_dir, "data/scene_detection_test")

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)

        # Skip non-image files
        if not os.path.isfile(fpath) or not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        try:
            preds = predict_image(fpath, topk=3)
            print(f"\nImage: {fname}")
            for label, prob in preds:
                print(f"  {label}: {prob:.4f}")
        except Exception as e:
            print(f"Error with {fpath}: {e}")