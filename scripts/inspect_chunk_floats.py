import os
import requests
import zstandard as zstd
import struct

def download_and_inspect():
    url = "https://storage.googleapis.com/lczero/training/data/t80-5/20240101-120000.zst"
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print("Failed to download: ", response.status_code)
        return
        
    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(response.raw)
    
    # Read the first chunk record
    version_bytes = stream_reader.read(4)
    version = int.from_bytes(version_bytes, "little", signed=False)
    
    record_size = 8356 # v6
        
    rest = stream_reader.read(record_size - 4)
    record = version_bytes + rest
    
    # Parse floats
    stm_offset = 4 + 1858 * 4 + 104 * 8 + 4
    floats_offset = stm_offset + 4
    
    # 15 floats = 60 bytes
    floats_bytes = record[floats_offset:floats_offset + 60]
    floats = struct.unpack("<15f", floats_bytes)
    
    print(f"Version: {version}")
    print(f"Floats: {floats}")
    
if __name__ == "__main__":
    download_and_inspect()
