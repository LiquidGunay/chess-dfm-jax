#!/usr/bin/env python3
import os
import time
import requests
import subprocess
from bs4 import BeautifulSoup # Need a simple way to parse if possible, or regex
import re

BASE_URL = "https://storage.lczero.org/files/training_data/test80/"
GCS_BUCKETS = [
    bucket.strip()
    for bucket in os.environ.get(
        "CHESS_DFM_CHUNK_BUCKETS",
        "gs://your-us-central1-bucket/data/chunks,gs://your-us-central2-bucket/data/chunks",
    ).split(",")
    if bucket.strip()
]

def get_links():
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            print("Server is up. Looking for .tar or .zst files...")
            links = re.findall(r'href="([^"]+\.(?:tar|zst))"', response.text)
            return links
        else:
            print(f"Server returned status {response.status_code}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    return []

def main():
    print(f"Polling {BASE_URL} for training data...")
    downloaded = 0
    target_downloads = 2 # Download a couple of files

    while downloaded < target_downloads:
        links = get_links()
        if links:
            # Sort to get the latest or just take the first few
            # the links might be relative
            for link in links[:target_downloads]:
                file_url = BASE_URL + link if not link.startswith('http') else link
                filename = link.split('/')[-1]
                local_path = f"/tmp/{filename}"
                
                print(f"Downloading {file_url} to {local_path}...")
                try:
                    with requests.get(file_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(local_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"Download complete: {local_path}")
                    
                    # If it's a tar file, we might want to extract it, but if it contains .zst chunks
                    # we can upload the extracted .zst chunks. Or we upload the .tar if our loader supports it?
                    # The data loader discovers .zst and .gz. If it's .tar, we need to extract.
                    if local_path.endswith(".tar"):
                        extract_dir = f"/tmp/extracted_{filename}"
                        os.makedirs(extract_dir, exist_ok=True)
                        print(f"Extracting {local_path} to {extract_dir}...")
                        subprocess.run(["tar", "-xf", local_path, "-C", extract_dir], check=True)
                        
                        # Upload extracted .zst files
                        for root, _, files in os.walk(extract_dir):
                            for f in files:
                                if f.endswith(".zst"):
                                    extracted_file = os.path.join(root, f)
                                    for bucket in GCS_BUCKETS:
                                        print(f"Uploading {f} to {bucket}...")
                                        subprocess.run(["/snap/google-cloud-cli/current/bin/gcloud", "storage", "cp", extracted_file, f"{bucket}/{f}"], check=True)
                        
                        # Clean up
                        subprocess.run(["rm", "-rf", extract_dir])
                        os.remove(local_path)
                        downloaded += 1
                        
                    elif local_path.endswith(".zst"):
                        for bucket in GCS_BUCKETS:
                            print(f"Uploading {filename} to {bucket}...")
                            subprocess.run(["/snap/google-cloud-cli/current/bin/gcloud", "storage", "cp", local_path, f"{bucket}/{filename}"], check=True)
                        os.remove(local_path)
                        downloaded += 1
                        
                except Exception as e:
                    print(f"Failed to process {file_url}: {e}")
                    
        if downloaded >= target_downloads:
            print("Successfully downloaded and uploaded target number of chunks. Exiting.")
            break
            
        print("Waiting 10 minutes before next poll...")
        time.sleep(600)

if __name__ == "__main__":
    main()
