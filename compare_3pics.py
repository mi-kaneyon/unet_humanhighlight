import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from unet import UNet  # 現在のunet.pyにあるUNetクラス(引数なし)をインポート
from PIL import Image

def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_mask(model, image, device):
    # 学習時と同じ正規化を適用 (3チャネル用)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    mask = (output > 0.5).astype(np.uint8) * 255
    return mask

def compare_masks(input_image_path, ground_truth_mask_path, model, device):
    # 入力画像と真値マスクの読み込み
    # データ収集時点で512x512で保存しているなら、そのまま読み込めばよい
    input_image = Image.open(input_image_path).convert("RGB")
    ground_truth_mask = Image.open(ground_truth_mask_path).convert("L")

    # モデルによる予測マスク生成
    predicted_mask = generate_mask(model, input_image, device)

    # 結果をプロット
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    
    # モデルのパスと画像パスの設定
    # dataフォルダがスクリプトと同階層、もしくは整合性が取れているか確認してください。
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'unet_model.pth')
    image_dir = os.path.join(base_dir, 'data', 'images')
    mask_dir = os.path.join(base_dir, 'data', 'masks')
    
    model = load_model(model_path, device)

    image_paths = sorted(os.listdir(image_dir))
    mask_paths = sorted(os.listdir(mask_dir))

    for image_name, mask_name in zip(image_paths, mask_paths):
        input_image_path = os.path.join(image_dir, image_name)
        ground_truth_mask_path = os.path.join(mask_dir, mask_name)
        compare_masks(input_image_path, ground_truth_mask_path, model, device)

if __name__ == "__main__":
    main()
