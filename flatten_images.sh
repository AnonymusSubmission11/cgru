#!/bin/bash

# Script to flatten the fid_generated_images directory structure
# Moves all images from subdirectories to the main fid_generated_images folder

BASE_DIR="/root/rl-machine-unlearning/fid_generated_images"

echo "Flattening directory structure in $BASE_DIR"
echo "Moving all images from subdirectories to main folder..."

# Count total images before processing
total_images=0
for subdir in "$BASE_DIR"/*/; do
    if [ -d "$subdir" ]; then
        image_count=$(find "$subdir" -name "*.png" | wc -l)
        total_images=$((total_images + image_count))
    fi
done

echo "Found $total_images images to move"

# Move all PNG files from subdirectories to main directory
moved_count=0
for subdir in "$BASE_DIR"/*/; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        echo "Processing directory: $subdir_name"
        
        # Move all PNG files from this subdirectory to main directory
        for image in "$subdir"/*.png; do
            if [ -f "$image" ]; then
                filename=$(basename "$image")
                mv "$image" "$BASE_DIR/$filename"
                moved_count=$((moved_count + 1))
                if [ $((moved_count % 100)) -eq 0 ]; then
                    echo "  Moved $moved_count/$total_images images..."
                fi
            fi
        done
        
        # Remove empty subdirectory
        rmdir "$subdir" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  Removed empty directory: $subdir_name"
        else
            echo "  Warning: Could not remove directory $subdir_name (may not be empty)"
        fi
    fi
done

echo "Flattening complete!"
echo "Moved $moved_count images total"
echo "All images are now in: $BASE_DIR"
