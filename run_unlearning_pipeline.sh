#!/bin/bash

# Comprehensive pipeline script for unlearning training and evaluation
# Usage: ./run_unlearning_pipeline.sh <class_name>
# Example: ./run_unlearning_pipeline.sh Cats

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if class name is provided
if [ $# -eq 0 ]; then
    print_error "Usage: $0 <class_name>"
    print_error "Example: $0 Cats"
    exit 1
fi

CLASS_NAME="$1"

# Define available classes (must match the list in generate_images_for_metrics.py)
AVAILABLE_CLASSES=("Architectures" "Bears" "Birds" "Butterfly" "Cats" "Dogs" "Fishes" "Flame" "Flowers" "Frogs" "Horses" "Human" "Jellyfish" "Rabbits" "Sandwiches" "Sea" "Statues" "Towers" "Trees" "Waterfalls")

# Validate class name
CLASS_VALID=false
for class in "${AVAILABLE_CLASSES[@]}"; do
    if [ "$CLASS_NAME" = "$class" ]; then
        CLASS_VALID=true
        break
    fi
done

if [ "$CLASS_VALID" = false ]; then
    print_error "Invalid class name: $CLASS_NAME"
    print_error "Available classes: ${AVAILABLE_CLASSES[*]}"
    exit 1
fi

print_success "Valid class name: $CLASS_NAME"

# Set up paths and variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$PROJECT_ROOT/config"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/training/train.py"
GENERATE_SCRIPT="$PROJECT_ROOT/scripts/testing/generate_images_for_metrics.py"
FLATTEN_SCRIPT="$PROJECT_ROOT/flatten_images.sh"
CLASSIFICATION_SCRIPT="$PROJECT_ROOT/classification.py"

# Check if required scripts exist
print_status "Checking required files..."

for script in "$TRAIN_SCRIPT" "$GENERATE_SCRIPT" "$FLATTEN_SCRIPT" "$CLASSIFICATION_SCRIPT"; do
    if [ ! -f "$script" ]; then
        print_error "Required script not found: $script"
        exit 1
    fi
done

print_success "All required scripts found"

# Create dynamic config file based on ac_cat2.py
CONFIG_NAME="ac_${CLASS_NAME,,}_unlearning.py"  # Convert to lowercase
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"

print_status "Creating dynamic config for $CLASS_NAME unlearning based on ac_cat2.py..."

cat > "$CONFIG_PATH" << EOF
import ml_collections
from config.base import get_config as get_base_config


def get_config():
    """Configuration for using the Actor-Critic method with trained style critic for ${CLASS_NAME} unlearning"""
    config = get_base_config()  # Start with base config
    
    # Enable Actor-Critic method
    config.use_actor_critic = True
    config.critic_fn = "cat_clip_scorer"  # Use the trained CLIP critic
    config.critic_fn_kwargs = {"class_to_remove": "${CLASS_NAME}"}  # Pass class to remove
    
    # Set logging and run name
    config.logdir = "logs/ac_${CLASS_NAME,,}_unlearning"
    config.run_name = "ddpo_ac_${CLASS_NAME,,}_unlearning"
    
    # Use same function for reward (critic and reward should be the same)
    config.reward_fn = "cat_clip_unlearning"
    config.reward_fn_kwargs = {"target_class_name": "${CLASS_NAME}"}
    
    config.train.learning_rate = 3e-4
    config.train.batch_size = 2  # Increased for 48GB GPU
    config.train.gradient_accumulation_steps = 4  # Increased for better gradient estimates
    
    # Enable time weighting for AC method (default beta=0.9)
    config.train.time_weighting_beta = 0.0
    # Disable advantage normalization by default
    config.train.normalize_advantages = True

    # Sampling configuration optimized for 48GB GPU
    config.sample.batch_size = 4  # Increased for 48GB GPU
    config.sample.num_batches_per_epoch = 4  # More samples per epoch
    config.sample.num_steps = 50  # Increased from 20 to 30 steps
    
    # Training epochs
    config.num_epochs = 101

    config.prompt_fn = "${CLASS_NAME,,}_clip_dataset"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
        
    return config
EOF

print_success "Config created: $CONFIG_PATH"

# Step 1: Train the model
print_status "Step 1: Training model for $CLASS_NAME unlearning..."
print_status "This may take several hours depending on your hardware..."

cd "$PROJECT_ROOT"
accelerate launch "$TRAIN_SCRIPT" --config="$CONFIG_PATH"

if [ $? -ne 0 ]; then
    print_error "Training failed!"
    exit 1
fi

print_success "Training completed successfully!"

# Step 2: Find the latest checkpoint
print_status "Step 2: Finding latest checkpoint..."

LOG_DIR="$PROJECT_ROOT/logs/ac_${CLASS_NAME,,}_unlearning"
if [ ! -d "$LOG_DIR" ]; then
    print_error "Log directory not found: $LOG_DIR"
    exit 1
fi

# Find the most recent run directory
LATEST_RUN=$(ls -t "$LOG_DIR" | head -n1)
if [ -z "$LATEST_RUN" ]; then
    print_error "No run directories found in $LOG_DIR"
    exit 1
fi

RUN_DIR="$LOG_DIR/$LATEST_RUN"
print_status "Using run directory: $RUN_DIR"

# Find the latest checkpoint
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    print_error "Checkpoints directory not found: $CHECKPOINT_DIR"
    exit 1
fi

LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR" | grep "checkpoint_" | head -n1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    print_error "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

CHECKPOINT_PATH="$CHECKPOINT_DIR/$LATEST_CHECKPOINT"
print_success "Found latest checkpoint: $CHECKPOINT_PATH"

# Step 3: Generate images using the trained model
print_status "Step 3: Generating images using trained model..."

# Update the generate_images_for_metrics.py script to use the checkpoint
python -c "
import sys
sys.path.append('$PROJECT_ROOT')
import re

# Read the current script
with open('$GENERATE_SCRIPT', 'r') as f:
    content = f.read()

# Update the checkpoint path
new_content = re.sub(
    r'CHKP = \".*\"',
    'CHKP = \"$CHECKPOINT_PATH\"',
    content
)

# Write back the updated script
with open('$GENERATE_SCRIPT', 'w') as f:
    f.write(new_content)

print('Updated checkpoint path in generate_images_for_metrics.py')
"

python "$GENERATE_SCRIPT"

if [ $? -ne 0 ]; then
    print_error "Image generation failed!"
    exit 1
fi

print_success "Image generation completed successfully!"

# Step 4: Flatten the directory structure
print_status "Step 4: Flattening directory structure..."

# Update the flatten_images.sh script to use the correct path
sed -i "s|BASE_DIR=\"/root/rl-machine-unlearning/fid_generated_images\"|BASE_DIR=\"$PROJECT_ROOT/fid_generated_images\"|g" "$FLATTEN_SCRIPT"

bash "$FLATTEN_SCRIPT"

if [ $? -ne 0 ]; then
    print_error "Directory flattening failed!"
    exit 1
fi

print_success "Directory structure flattened successfully!"

# Step 5: Run classification on generated images
print_status "Step 5: Running classification on generated images..."

INPUT_DIR="$PROJECT_ROOT/fid_generated_images"
OUTPUT_DIR="$PROJECT_ROOT/results/${CLASS_NAME,,}_classification_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

python "$CLASSIFICATION_SCRIPT" \
    --input_dir="$INPUT_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --ckpt="ddpo_pytorch/assets/clip_probe_20cls_ep4.pt"

if [ $? -ne 0 ]; then
    print_error "Classification failed!"
    exit 1
fi

print_success "Classification completed successfully!"

# Final summary
print_success "=== PIPELINE COMPLETED SUCCESSFULLY ==="
print_status "Class: $CLASS_NAME"
print_status "Config used: $CONFIG_PATH"
print_status "Checkpoint used: $CHECKPOINT_PATH"
print_status "Generated images: $INPUT_DIR"
print_status "Classification results: $OUTPUT_DIR"
print_status "Log directory: $RUN_DIR"

print_success "All steps completed successfully!"
