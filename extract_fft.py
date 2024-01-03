import torch
import torch.fft as fft
from PIL import Image
from torchvision import transforms, utils

def extract_frequency_pytorch_pil(image_path):
    # Load the image using Pillow
    img = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img)

    # Apply 2D Fourier Transform
    f_transform = fft.fftn(img_tensor, dim=(-2, -1))
    f_transform_shifted = fft.fftshift(f_transform, dim=(-2, -1))

    # Get image dimensions
    _, rows, cols = img_tensor.shape

    # Create a mask for high frequencies
    mask_high = torch.ones((rows, cols), dtype=torch.float32)
    r_center, c_center = rows // 2, cols // 2
    r_cutoff, c_cutoff = 30, 30  # Adjust these values for high-pass filtering
    mask_high[r_center - r_cutoff:r_center + r_cutoff,
              c_center - c_cutoff:c_center + c_cutoff] = 0

    # Apply the high-pass filter
    f_transform_shifted_high = f_transform_shifted * mask_high

    # Inverse Fourier Transform to get the high-frequency image
    img_high = fft.ifftn(fft.ifftshift(f_transform_shifted_high, dim=(-2, -1)), dim=(-2, -1)).real

    # Create a mask for low frequencies
    mask_low = 1 - mask_high

    # Apply the low-pass filter
    f_transform_shifted_low = f_transform_shifted * mask_low

    # Inverse Fourier Transform to get the low-frequency image
    img_low = fft.ifftn(fft.ifftshift(f_transform_shifted_low, dim=(-2, -1)), dim=(-2, -1)).real

    return img_low, img_high

def restore_image(img_low, img_high):
    # Combine low and high frequencies
    first_high = img_high[:, 2:258, :]
    for i in range(img_high.size(1)//256):
        img_high[:, 2*(i+1)+256*i: (i+1)*258, :] = first_high
    restored_img = img_low + img_high

    # Clip values to ensure they are in the valid intensity range [0, 1]
    restored_img = torch.clamp(restored_img, 0, 1)

    return restored_img

if __name__ == "__main__":
    # Load the image and extract low/high frequencies
    image_path = "saved_info/2_pointers_consistency/celeba_256/first_exp_0.1_eps_1k_weighted_fourier_loss_t_fix_combine/x0_iter_6900.png"
    img_low, img_high = extract_frequency_pytorch_pil(image_path)

    # Restore the image
    img_restored = restore_image(img_low, img_high)

    # Save the original image, low and high frequency images, and the restored image
    transforms.ToPILImage()(img_low).save("low_frequencies.png")
    transforms.ToPILImage()(img_high).save("high_frequencies.png")
    transforms.ToPILImage()(img_restored).save("restored_image.png")

    print("Images saved successfully.")

