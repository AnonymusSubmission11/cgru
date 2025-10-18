import os
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from PIL import Image
import io

# Path to your downloaded dataset folder
data_dir = "/root/.cache/huggingface/hub/datasets--OPTML-Group--UnlearnCanvas/snapshots/71b565de9ed3401986c62c04352e51ba32d79fb0/data"   # change this!

# Output folder for extracted images
out_dir = "unlearncanvas_images"
os.makedirs(out_dir, exist_ok=True)

x = 0
# Iterate through parquet files
for file in os.listdir(data_dir):
    if file.endswith(".parquet"):
        parquet_path = os.path.join(data_dir, file)
        print(f"Processing {parquet_path} ...")

        # Load parquet
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Check available columns
        print("Columns:", df.columns)

        # Assuming the image column is "image" or "image_bytes"
        for i, row in df.iterrows():
            #if "image" in df.columns:  # already decoded in HF style
            row = row["image"]
            #elif "image_bytes" in df.columns:  # stored as bytes
            img = Image.open(io.BytesIO(row["bytes"]))
            #else:
             #   raise ValueError("❌ No image column found)!")
            #print(list(img.keys()))
            img.save(os.path.join(out_dir, str(x) + ".jpg"))
            x += 1

        # Delete parquet to save space
        os.remove(parquet_path)
        print(f"✅ Finished {file}, deleted parquet.")

print("🎉 Done! All images extracted.")
