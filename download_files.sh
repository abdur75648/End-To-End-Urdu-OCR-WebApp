#!/bin/bash

# URLs of the files to be downloaded
urls=(
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/1.jpg"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/2.jpg"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/3.jpg"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/best_norm_ED.pth"
  "https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/yolov8m_UrduDoc.pt"
)

# Download each file
for url in "${urls[@]}"; do
  echo "Downloading $url..."
  curl -L -o $(basename "$url") "$url"
done

echo "All files downloaded successfully."
