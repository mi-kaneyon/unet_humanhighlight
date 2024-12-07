import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image

def load_segmentation_model(device):
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    )
    model.to(device)
    model.eval()
    return model

def perform_segmentation(model, image, device):
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    if 'out' in outputs:
        segmentation_mask = torch.argmax(outputs['out'][0], dim=0).byte().cpu().numpy()
    else:
        segmentation_mask = torch.argmax(outputs[0], dim=0).byte().cpu().numpy()

    person_mask = (segmentation_mask == 15).astype(np.uint8) * 255
    return person_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_segmentation_model(device)

    images_dir = 'data/images'
    masks_dir = 'data/masks'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_count = 0
    target_size = (512, 512)  # 512x512に変更する場合

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        person_mask = perform_segmentation(model, frame, device)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_mask = Image.fromarray(person_mask, mode='L')

        pil_image_resized = pil_image.resize(target_size, Image.BILINEAR)
        pil_mask_resized = pil_mask.resize(target_size, Image.NEAREST)

        image_path = os.path.join(images_dir, f'frame_{frame_count:05d}.png')
        mask_path = os.path.join(masks_dir, f'frame_{frame_count:05d}_mask.png')

        pil_image_resized.save(image_path)
        pil_mask_resized.save(mask_path)

        frame_count += 1
        print(f"Saved frame {frame_count}")

        frame_seg = cv2.bitwise_and(cv2.cvtColor(np.array(pil_image_resized), cv2.COLOR_RGB2BGR),
                                    cv2.cvtColor(np.array(pil_image_resized), cv2.COLOR_RGB2BGR),
                                    mask=np.array(pil_mask_resized))
        cv2.imshow('Data Collection - Press q to quit', frame_seg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
