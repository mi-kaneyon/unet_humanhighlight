import os
import cv2
import torch
import torchvision.transforms as transforms
from unet import UNet  # 学習時と同じUNet定義を使用
from PIL import Image
import numpy as np

def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def perform_segmentation(model, image, device):
    # 学習時と同じ正規化を適用 (512x512想定)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    # 出力を二値化
    mask = (output > 0.5).astype(np.uint8) * 255
    return mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 学習済みモデルのパス
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'models', 'unet_model.pth')

    model = load_model(model_path, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    target_size = (512, 512)  # 学習時と同様に512x512を使用

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        original_height, original_width = frame.shape[:2]

        # フレームを512x512にリサイズしてPIL画像に変換
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(target_size)

        # セグメンテーション実行（maskは512x512）
        mask = perform_segmentation(model, pil_image, device)

        # 推論後、maskをオリジナルフレームサイズに戻す
        mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # マスク適用
        frame_seg = cv2.bitwise_and(frame, frame, mask=mask_resized)

        cv2.imshow('Real-Time Segmentation - Press q to quit', frame_seg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
