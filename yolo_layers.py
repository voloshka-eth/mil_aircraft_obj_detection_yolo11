import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

yolo = YOLO("jet_classifier/yolov11_detect4/weights/best.pt")
model = yolo.model.eval()

img_path = r"C:\Users\ivank\Desktop\f126483bbb7ae8cb.jpg"
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((960, 960)),
    transforms.ToTensor(),
])

input_tensor = transform(img).unsqueeze(0)

conv_layers = []
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d):
        conv_layers.append((name, layer))

print(f"Found Conv layers: {len(conv_layers)}")

features = {}

def make_hook(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

hooks = []
for name, layer in conv_layers:
    hooks.append(layer.register_forward_hook(make_hook(name)))

with torch.no_grad():
    _ = model(input_tensor)

for h in hooks:
    h.remove()

for name, fmap in features.items():
    fmap = fmap[0]  # (C,H,W)
    C = fmap.shape[0]

    print(f"{name} -> shape: {fmap.shape}")

    n = min(16, C)
    plt.figure(figsize=(6,6))
    for i in range(n):
        plt.subplot(4,4,i+1)
        plt.imshow(fmap[i].cpu(), cmap="gray")
        plt.axis("off")

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()
