"""
Script to edit neural_activity_grid.gif:
1. Keep only first 800 frames
2. Recolor for white background
"""

from PIL import Image
import numpy as np

def recolor_for_white_background(frame):
    """
    Recolor a frame to look good on white background.
    Inverts dark/neutral backgrounds to light, but preserves reds and blues.
    """
    # Convert to RGBA if not already
    if frame.mode != 'RGBA':
        frame = frame.convert('RGBA')

    arr = np.array(frame, dtype=np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3:4]

    # Calculate color properties for each pixel
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Detect "colorful" pixels (reds and blues) vs neutral/gray pixels
    # A pixel is colorful if one channel dominates the others
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    saturation = (max_channel - min_channel) / (max_channel + 1e-6)

    # Identify red pixels: high R, low G and B
    is_red = (r > g + 30) & (r > b + 30) & (saturation > 0.3)

    # Identify blue pixels: high B, low R and G
    is_blue = (b > r + 30) & (b > g + 30) & (saturation > 0.3)

    # Mask for colored pixels (reds and blues) - these become blue
    colored_mask = is_red | is_blue

    # Invert non-colored pixels (background/neutral colors)
    inverted_rgb = 255 - rgb

    # Define the target blue color
    blue_color = np.array([30, 100, 200], dtype=np.float32)  # A nice blue

    # Start with inverted background
    result_rgb = inverted_rgb.copy()

    # Replace colored pixels with blue
    colored_mask_3d = np.stack([colored_mask] * 3, axis=2)
    result_rgb = np.where(colored_mask_3d, blue_color, result_rgb)

    # Combine with alpha
    result = np.concatenate([result_rgb, alpha], axis=2)

    return Image.fromarray(result.astype('uint8'), 'RGBA')

def process_gif(input_path, output_path, max_frames=800):
    """Process the gif: trim frames and recolor."""
    print(f"Opening {input_path}...")
    img = Image.open(input_path)

    frames = []
    durations = []

    try:
        frame_count = 0
        while frame_count < max_frames:
            # Get frame duration
            duration = img.info.get('duration', 100)
            durations.append(duration)

            # Process frame
            frame = img.copy()
            recolored = recolor_for_white_background(frame)
            frames.append(recolored)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

            # Move to next frame
            img.seek(img.tell() + 1)

    except EOFError:
        print(f"Reached end of gif at frame {frame_count}")

    print(f"Total frames to save: {len(frames)}")

    # Convert RGBA frames to P mode (palette) for gif saving
    # Use the first frame's palette as reference
    processed_frames = []
    for frame in frames:
        # Convert to RGB with white background
        rgb_frame = Image.new('RGB', frame.size, (255, 255, 255))
        rgb_frame.paste(frame, mask=frame.split()[3] if frame.mode == 'RGBA' else None)
        processed_frames.append(rgb_frame)

    # Save
    print(f"Saving to {output_path}...")
    processed_frames[0].save(
        output_path,
        save_all=True,
        append_images=processed_frames[1:],
        duration=durations[0] if durations else 100,
        loop=0
    )
    print("Done!")

if __name__ == "__main__":
    input_file = "neural_activity_grid.gif"
    output_file = "neural_activity_grid_edited.gif"
    process_gif(input_file, output_file, max_frames=800)
