import requests
import os
import time

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        # Check if content is HTML (often happens with wrong raw links)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"Error: URL {url} returned HTML content, not a binary file.")
            return False
            
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        wrote = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    wrote += len(chunk)
                    # print(f"\rDownloaded {wrote / 1024 / 1024:.2f} MB", end="")
        
        print(f"\nDownloaded {filename} successfully ({wrote / 1024 / 1024:.2f} MB).")
        return True
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename) # Remove partial file
        return False

if __name__ == "__main__":
    # Ensure weights directory exists
    if not os.path.exists("weights"):
        os.makedirs("weights")

    # 1. Download YOLOv8n
    yolo_urls = [
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    ]
    
    if not os.path.exists("yolov8n.pt") or os.path.getsize("yolov8n.pt") < 1024*1024:
        print("Checking yolov8n.pt...")
        for url in yolo_urls:
            if download_file(url, "yolov8n.pt"):
                break
            time.sleep(1)
    else:
        print("yolov8n.pt already exists and seems valid.")

    # 2. Download TrackNet weights
    # Note: The raw link should point to the actual binary file.
    # If this fails, user might need to download manually from Google Drive or similar.
    tracknet_url = "https://github.com/ChgygLin/TrackNetV2-pytorch/raw/main/tf2torch/track.pt"
    tracknet_path = "weights/track.pt"
    
    if os.path.exists(tracknet_path):
        # Check if it's the invalid HTML file (small size)
        if os.path.getsize(tracknet_path) < 10 * 1024 * 1024:
            print(f"Found invalid (small) {tracknet_path}. Deleting and re-downloading...")
            os.remove(tracknet_path)
    
    if not os.path.exists(tracknet_path):
        print(f"Downloading TrackNet weights to {tracknet_path}...")
        if not download_file(tracknet_url, tracknet_path):
            print("Failed to download TrackNet weights automatically.")
            print("Please manually download 'track.pt' from https://github.com/ChgygLin/TrackNetV2-pytorch/blob/main/tf2torch/track.pt")
            print("Make sure to click the 'Download raw file' button or use the direct link correctly.")
            print("Then place it in the 'weights' folder.")
    else:
        print(f"{tracknet_path} already exists.")
