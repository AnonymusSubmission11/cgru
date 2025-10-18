from PIL import Image
import io
import numpy as np
import torch
import torch.nn as nn
import os
from transformers import CLIPProcessor
from torchvision import transforms as T
import torch.nn.functional as F
import pickle
import clip


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def cat_likeness():
    """Cat-likeness reward function based on COCO cat embeddings
    
    Returns scores from 0 to 10, where:
    - 0-2: Not cat-like at all
    - 3-5: Somewhat cat-like
    - 6-8: Very cat-like
    - 9-10: Extremely cat-like
    """
    from transformers import CLIPModel, CLIPProcessor
    
    # Load pre-computed cat embeddings
    cat_embeddings_path = "ddpo_pytorch/assets/cat_embeddings.pt"
    if not os.path.exists(cat_embeddings_path):
        raise FileNotFoundError(f"Cat embeddings not found at {cat_embeddings_path}. Run prepare_cat_embeddings.py first.")
    
    cat_embeddings = torch.load(cat_embeddings_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cat_embeddings = cat_embeddings.to(device)
    
    # Load CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Use a fixed threshold for more aggressive discrimination
    # This threshold is based on typical CLIP cosine similarities
    similarity_threshold = 0.25  # Images above this are considered cat-like
    print(f"Using similarity threshold: {similarity_threshold}")
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            # Convert tensor to PIL images
            if images.dim() == 4:  # (batch, channels, height, width)
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
                images = images.permute(0, 2, 3, 1).cpu().numpy()  # NCHW -> NHWC
            pil_images = [Image.fromarray(img) for img in images]
        else:
            pil_images = images
        
        # Process with CLIP
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / torch.linalg.vector_norm(image_features, dim=-1, keepdim=True)
        
        # Compute similarity with cat embeddings
        similarities = torch.mm(image_features, cat_embeddings.T)  # (batch_size, num_cats)
        
        mean_similarities = similarities.mean(dim=1)  # (batch_size,)
        
        # More aggressive threshold-based scoring
        # Use a steep sigmoid around the threshold for sharp discrimination
        temperature = 5.0  # Very high temperature for sharp transition
        centered_similarities = mean_similarities - similarity_threshold
        sigmoid_scores = torch.sigmoid(centered_similarities * temperature)
        
        # Scale to [0, 10] range, then reverse for unlearning (lower cat-likeness = higher reward)
        cat_scores = sigmoid_scores * 10.0
        reversed_scores = 10.0 - cat_scores  # Reverse: 0 becomes 10, 10 becomes 0
        return reversed_scores, {}

    return _fn

def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


class CriticWrapper:
    """Wrapper class to hold the critic and provide the function interface"""
    def __init__(self, critic):
        self.critic = critic
    
    def __call__(self, images, timesteps, prompts, metadata):
        """
        Critic function that predicts value functions.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width) - decoded images
            timesteps: Tensor of shape (batch_size,) - diffusion timesteps  
            prompts: List of prompts
            metadata: Additional metadata
            
        Returns:
            values: Tensor of shape (batch_size,) - predicted value functions
            metadata: Additional metadata
        """
        # NOTE: Use torch.no_grad() because critic values are only used as scalar weights
        # They should not participate in backpropagation through the diffusion model
        with torch.no_grad():
            values = self.critic(images, timesteps, prompts)
        return values, metadata

class DummyCritic(nn.Module):
    """
    A dummy critic that predicts value functions for noised images.
    This is a placeholder that can be replaced with a real critic.
    The critic operates on decoded images from latents and predicts expected terminal reward.
    """
    
    def __init__(self, image_size=512, num_channels=3):
        super().__init__()
        # Much more aggressive downsampling to reduce memory
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, stride=8),   # 512x512 -> 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=4),            # 64x64 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=4),           # 16x16 -> 4x4
            nn.ReLU(),
        )
        
        # Much smaller value prediction head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),  # Only 2048 -> 64 instead of 921600 -> 512
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    
    def forward(self, images, timesteps=None, prompts=None):
        """
        Forward pass of the critic.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width) - decoded images
            timesteps: Optional tensor of shape (batch_size,) - diffusion timesteps
            prompts: Optional list of prompts
            
        Returns:
            values: Tensor of shape (batch_size,) - predicted value functions
        """
        # Ensure images are in the same dtype as the model parameters
        if images.dtype != next(self.parameters()).dtype:
            images = images.to(dtype=next(self.parameters()).dtype)
        
        features = self.conv_layers(images)
        values = self.value_head(features).squeeze(-1)
        return values

def dummy_critic():
    """ 
    Returns a dummy critic function that can be used in place of a real critic.
    The critic predicts value functions for noised images.
    """
    critic = DummyCritic()
    return CriticWrapper(critic)


def aesthetic_critic():
    """ 
    Returns a critic function that uses the trained aesthetic critic.
    The critic predicts value functions for noised images.
    """
    import os
    
    # Import CriticModel from standalone module to avoid datasets dependency
    from critic_model import CriticModel

    # Check if the trained critic exists
    critic_path = "critic_models/aesthethic/aesthethic_critic.pth"
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"Trained critic not found at {critic_path}. Please train the critic first using critic_training/train_critic.py")
    
    # Load the trained critic
    critic = CriticModel()
    state_dict = torch.load(critic_path, map_location='cpu')
    
    # Load state dict with strict=False to handle missing keys (like position_ids)
    missing_keys, unexpected_keys = critic.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
    critic.eval()
    
    print(f"Loaded trained aesthetic critic from {critic_path}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    return CriticWrapper(critic)


def cat_likeness_critic():
    """Cat-likeness critic function for Actor-Critic training"""
    import os
    
    # Import CriticModel from standalone module to avoid datasets dependency
    from critic_model import CriticModel

    # Check if the trained critic exists
    critic_path = "critic_training/output/cat_likeness_critic.pt"
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"Trained critic not found at {critic_path}. Please train the critic first using critic_training/train_critic.py")
    
    # Load the trained critic
    critic = CriticModel()
    state_dict = torch.load(critic_path, map_location='cpu')
    
    # Load state dict with strict=False to handle missing keys (like position_ids)
    missing_keys, unexpected_keys = critic.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
    critic.eval()
    
    print(f"Loaded trained aesthetic critic from {critic_path}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    return CriticWrapper(critic)

def books_unlearning():
    """
    Books unlearning reward function using YOLOv8 detector.
    Returns 1 - confidence that 'books' are detected, multiplied by 10.
    Higher scores = less books detected = better for unlearning.
    """
    from ultralytics import YOLO
    import torch
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano version for speed
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            # Convert from tensor to PIL images
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
            pil_images = [Image.fromarray(img) for img in images]
        else: 
            pil_images = images
        
        # Get predictions using YOLOv8
        books_confidences = []
        
        for pil_image in pil_images:
            # Run YOLOv8 inference
            results = model(pil_image, verbose=False)
            
            # Look for 'book' class (COCO class 73)
            book_confidences = []
        
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # COCO class 73 is 'book'
                        if class_id == 73:
                            book_confidences.append(confidence)    
            # Get maximum confidence for books in this image
            if book_confidences:
                max_confidence = max(book_confidences)
            else:
                max_confidence = 0.0
            
            books_confidences.append(max_confidence)
        
        books_confidences = np.array(books_confidences)
        
        # Reward = (1 - confidence) * 10
        # Higher reward = less books detected = better for unlearning
        rewards = (1.0 - books_confidences) * 10.0
        
        return rewards, {}
    
    return _fn

def books_critic():
    """Books unlearning critic function for Actor-Critic training"""
    import os
    from scripts.testing.critic_model import CriticModel
    
    # Load the trained books critic
    critic_path = "critic_models/books/books_critic.pth"
    
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"Books critic not found at {critic_path}. Please train it first.")
    
    # Load the critic model
    critic = CriticModel(
        hidden_size=768,  # CLIP hidden size
        cross_attention_dim=768,
        num_timesteps=1000
    )
    
    # Load the trained weights
    checkpoint = torch.load(critic_path, map_location='cpu')
    critic.load_state_dict(checkpoint['model_state_dict'])
    
    if torch.cuda.is_available():
        critic = critic.cuda()
    
    critic.eval()
    
    print(f"Loaded trained books critic from {critic_path}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    return CriticWrapper(critic)


def cat_clip_unlearning(target_class_name: str = "Dogs"):
    """Cat unlearning reward function using CLIP classifier
    
    Uses log-odds transformation: log((1-p) / p) where p is cat probability.
    Returns scores from 0 to 10, where:
    - 0-2: Very cat-like (high confidence for cat class, p > 0.5)
    - 3-5: Somewhat cat-like (moderate confidence, p ≈ 0.5)
    - 6-8: Not very cat-like (low confidence, p < 0.5)
    - 9-10: Not cat-like at all (very low confidence, p << 0.5)
    
    The log-odds transformation provides better sensitivity around extreme probabilities.
    """
    from transformers import CLIPModel, CLIPProcessor
    import torch.nn as nn
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained CLIP classifier
    checkpoint_path = "ddpo_pytorch/assets/clip_probe_20cls_ep4.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CLIP classifier checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    cat_class_idx = class_names.index(target_class_name)
    
    # Load CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Freeze CLIP parameters
    for p in clip_model.parameters():
        p.requires_grad = False
    
    # Create classifier head
    class CosineClassifier(nn.Module):
        def __init__(self, in_dim: int, num_classes: int, init_logit_scale: float = 3.0):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
            nn.init.xavier_normal_(self.weight)
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))

        def forward(self, feats: torch.Tensor):
            feats = F.normalize(feats, dim=-1)
            W = F.normalize(self.weight, dim=-1)
            return feats @ W.t() * self.logit_scale.exp()
    
    classifier = CosineClassifier(
        in_dim=clip_model.vision_model.config.hidden_size,
        num_classes=len(class_names)
    )
    classifier.load_state_dict(checkpoint['classifier'])
    classifier = classifier.to(device)
    classifier.eval()
    
    print(f"Loaded CLIP classifier with {len(class_names)} classes")
    print(f"Cat class index: {cat_class_idx}")
    
    def _fn(images, prompts, metadata, timesteps=None):
        if isinstance(images, torch.Tensor):
            # Convert from tensor to PIL images
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(img) for img in images]
        else:
            pil_images = images
        
        rewards = []
        
        with torch.no_grad():
            for pil_image in pil_images:
                try:
                    # Process image with CLIP
                    inputs = processor(images=pil_image, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device)
                    
                    # Get CLIP features
                    vout = clip_model.vision_model(pixel_values=pixel_values)
                    feats = vout.pooler_output  # Use pooler output like in training
                    
                    # Get classifier predictions
                    logits = classifier(feats)
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get cat probability (class index 4)
                    cat_prob = probs[0, cat_class_idx].item()
                    
                    # Clip probability to avoid log(0) or log(inf)
                    cat_prob = np.clip(cat_prob, 1e-8, 1.0 - 1e-8)
                    
                    # Convert to log-odds: log(p / (1-p))
                    # For unlearning: we want lower cat probability = higher reward
                    # So we use log((1-p) / p) = -log(p / (1-p))
                    log_odds = np.log((1.0 - cat_prob) / cat_prob)
                    
                    # Scale to reasonable range (e.g., -5 to 5 maps to 0-10)
                    # log_odds typically ranges from about -6 to +6 for probabilities 0.002 to 0.998
                    reward = (log_odds + 6.0) / 12.0 * 10.0
                    reward = np.clip(reward, 0.0, 10.0)
                    rewards.append(reward)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    rewards.append(5.0)  # Neutral reward on error
        
        rewards = np.array(rewards)
        return rewards, {}
    
    return _fn

class CriticConceptWrapper:
    """Wrapper class to hold the critic and provide the function interface"""
    def __init__(self, backbone, clsfr, class_idx):
        self.backbone = backbone
        self.classifier = clsfr
        self.class_idx = class_idx
        self.resize_to = 224
        # self.to_tensor_01 = T.Compose([
        #     T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        #     T.CenterCrop(resize_to),
        #     T.ToTensor(),   # [0,1]
        # ])

        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.mean = torch.tensor(clip_proc.image_processor.image_mean, device="cuda").view(1, -1, 1, 1)
        self.std  = torch.tensor(clip_proc.image_processor.image_std,  device="cuda").view(1, -1, 1, 1)
    
    def __call__(self, images, timesteps, prompts, metadata):
        """
        Critic function that predicts value functions.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width) - decoded images
            timesteps: Tensor of shape (batch_size,) - diffusion timesteps  
            prompts: List of prompts
            metadata: Additional metadata
            
        Returns:
            values: Tensor of shape (batch_size,) - predicted value functions
            metadata: Additional metadata
        """
        # NOTE: Use torch.no_grad() because critic values are only used as scalar weights
        # They should not participate in backpropagation through the diffusion model
        rewards = []

        with torch.no_grad():
            images = images.to(dtype=next(self.backbone.parameters()).dtype)

            images = F.interpolate(images, size=(self.resize_to, self.resize_to),
                                   mode="bicubic", align_corners=False, antialias=True)
            images = (images - self.mean) / self.std

            timesteps = timesteps / 1000.0


            out = self.backbone(pixel_values=images, timesteps=timesteps)
            logits = self.classifier(out.pooler_output)


            probs = F.softmax(logits, dim=-1)
                    
            # Get cat probability (class index 4)
            cat_prob = probs[0, self.class_idx].item()
            
            # Clip probability to avoid log(0) or log(inf)
            cat_prob = np.clip(cat_prob, 1e-8, 1.0 - 1e-8)
            
            # Convert to log-odds: log(p / (1-p))
            # For unlearning: we want lower cat probability = higher reward
            # So we use log((1-p) / p) = -log(p / (1-p))
            log_odds = np.log((1.0 - cat_prob) / cat_prob)
            
            # Scale to reasonable range (e.g., -5 to 5 maps to 0-10)
            # log_odds typically ranges from about -6 to +6 for probabilities 0.002 to 0.998
            reward = (log_odds + 6.0) / 12.0 * 10.0
            reward = np.clip(reward, 0.0, 10.0)
            rewards.append(reward)
            rewards = np.array(rewards)
            
        return rewards, {}

def cat_clip_scorer(class_to_remove: str = "Cats"):
    """Books unlearning critic function for Actor-Critic training"""
    import os
    from timestep_aware_clip import TimestepAwareCLIPVisionModel
    from train_critic_clip import CosineClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained books critic
    critic_path = "timestep_aware_clip_ddim_vae.pt"
    state_dict = torch.load(critic_path)
    class_names = state_dict["class_names"]
    class_idx = class_names.index(class_to_remove)
    
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"Books critic not found at {critic_path}. Please train it first.")
    
    # Load the critic model
    backbone = TimestepAwareCLIPVisionModel.from_pretrained_clip(
        "openai/clip-vit-base-patch32",
        gate_add_one=True, time_embed_mult=4
    ).to(device)

    backbone.load_state_dict(state_dict["backbone_state_dict"], strict=False)

    feat_dim = backbone.config.hidden_size  # 768
    classifier = CosineClassifier(in_dim=feat_dim, num_classes=20).to(device)
    classifier.load_state_dict(state_dict["classifier_state_dict"])
    
    
    # if torch.cuda.is_available():
    #     critic = critic.cuda()
    
    # critic.eval()
    
    print(f"Loaded trained books critic from {critic_path}")
    
    return CriticConceptWrapper(backbone, classifier, class_idx=class_idx)


def van_gogh_style():
    """
    Van Gogh style similarity reward function based on reference implementation.
    Returns similarity score between input images and Van Gogh paintings.
    Higher scores = more Van Gogh-like style.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    import os
    import glob
    import numpy as np
    import random
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load VGG19 exactly like reference
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Normalization exactly like reference
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # Style layers exactly like reference
    style_layers_default = ['conv_1', 'conv_2', 'conv_3']
    
    # Image preprocessing exactly like reference
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    
    def image_loader(image):
        """Load and preprocess image exactly like reference"""
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        
        image_tensor = loader(pil_image).unsqueeze(0)
        return image_tensor.to(device, torch.float)
    
    def gram_matrix(input):
        """Gram matrix computation exactly like reference"""
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)  # normalize by number of elements
    
    class StyleLoss(nn.Module):
        """Style loss exactly like reference"""
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = target_feature.detach()
        
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
    
    class Normalization(nn.Module):
        """Normalization exactly like reference"""
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        
        def forward(self, img):
            return (img - self.mean) / self.std
    
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_features, style_layers):
        """Get style model exactly like reference"""
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        style_losses = []
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
            model.add_module(name, layer)
            
            if name in style_layers:
                target_feature = style_features[i-1].detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        
        # Trim off layers after last style loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        
        return model, style_losses
    
    # Load pre-computed Van Gogh style features
    features_path = "ddpo_pytorch/assets/van_gogh_style_features.pt"
    
    # Load pre-computed features
    print(f"Loading pre-computed Van Gogh features from {features_path}...")
    data = torch.load(features_path, map_location=device)
    all_van_gogh_features = data['features']
    style_layers = data['style_layers']
    
    print(f"Loaded {len(all_van_gogh_features)} Van Gogh paintings")
    
    print(f"Pre-computing style models for {len(all_van_gogh_features)} Van Gogh paintings...")
    
    # Pre-compute all style models once
    all_style_models = []
    with torch.no_grad():
        for van_gogh_features in all_van_gogh_features:
            # Create style model with pre-computed target features
            model, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, 
                                                            van_gogh_features, style_layers)
            
            all_style_models.append((model, style_losses))
    
    print(f"Pre-computed {len(all_style_models)} style models")
    
    def _fn(images, prompts, metadata, timesteps=None):
        if isinstance(images, torch.Tensor):
            # Convert from tensor to PIL images
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
            # Convert to PIL images
            pil_images = [Image.fromarray(img) for img in images]
        else:
            # Images are already PIL images
            pil_images = images
        
        rewards = []
        
        with torch.no_grad():
            for pil_image in pil_images:
                try:
                    # Load and preprocess input image
                    input_img = image_loader(pil_image)
                    
                    # Randomly sample 10 style models for this image
                    if len(all_style_models) > 10:
                        selected_models = random.sample(all_style_models, 10)
                    else:
                        selected_models = all_style_models
                    
                    # Compare with selected Van Gogh paintings
                    total_style_loss = 0.0
                    num_comparisons = 0
                    
                    for model, style_losses in selected_models:
                        # Compute style loss
                        model(input_img)
                        style_score = sum(sl.loss for sl in style_losses)
                        total_style_loss += style_score.item()
                        num_comparisons += 1
                    
                    # Calculate average style loss
                    avg_style_loss = total_style_loss / num_comparisons if num_comparisons > 0 else float('inf')
                    
                    # Convert loss to similarity score
                    # Lower style loss = more similar to Van Gogh = lower reward (we want to remove Van Gogh style)
                    # Higher style loss = less similar to Van Gogh = higher reward (we want to reward non-Van Gogh style)
                    similarity = 1.0 / (avg_style_loss + 1e-8)
                    
                    # Scale to 0-10 range
                    # Clip to reasonable range and scale
                    similarity = np.clip(similarity, 0, 100)  # Clip extreme values
                    scaled_similarity = (similarity / 100.0) * 10.0  # Scale to 0-10
                    rewards.append(scaled_similarity)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    rewards.append(0.0)
        
        rewards = np.array(rewards)
        
        return rewards, {}
    
    return _fn

def nsfw_unlearning():
    """
    NSFW unlearning reward function using HF NSFW classifier.
    Returns scores from 0-10 where higher = less NSFW (better for unlearning).
    """
    from transformers import pipeline
    
    # Load NSFW classifier
    classifier = pipeline("image-classification", 
                         model="Falconsai/nsfw_image_detection")
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)
            pil_images = [Image.fromarray(img) for img in images]
        else:
            pil_images = images
        
        rewards = []
        for pil_image in pil_images:
            # Get classification results (returns list of dicts)
            results = classifier(pil_image)
            
            # Find NSFW probability from results
            nsfw_prob = 0.0
            for result in results:
                if result['label'].lower() == 'nsfw':
                    nsfw_prob = result['score']
                    break
            
            # Apply scale [0-10]
            # reward = (1.0 - nsfw_prob) # * 10.0
            reward = np.log((1.0 - nsfw_prob + 1e-6) / (nsfw_prob + 1e-6))
            reward = np.tanh(0.3 * reward) * 10  # squash to [-10, +10] smoothly
            rewards.append(reward)
                
        return np.array(rewards), {}
    
    return _fn

def nsfw_q16_unlearning():
    """
    NSFW unlearning reward function using Q16 inappropriateness classifier.
    Returns scores from 0-10 where higher = less NSFW (better for unlearning).
    Uses the same Q16 model as HUB evaluation for consistency.
    """
    import pickle
    import clip
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Q16 prompts (same as HUB evaluation)
    q16_prompt_path = "../HUB/models/q16/prompts.p"
    if not os.path.exists(q16_prompt_path):
        raise FileNotFoundError(f"Q16 prompts not found at {q16_prompt_path}")
    
    trained_prompts = torch.HalfTensor(pickle.load(open(q16_prompt_path, "rb"))).to(device)
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-L/14", device, jit=False)
    clip_model.eval()
    
    # Create classifier
    class SimClassifier(nn.Module):
        def __init__(self, embeddings):
            super(SimClassifier, self).__init__()
            self.embeddings = nn.parameter.Parameter(embeddings)
        
        def forward(self, x):
            embeddings_norm = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)
            image_features_norm = x / x.norm(dim=-1, keepdim=True)
            similarity = 100.0 * image_features_norm @ embeddings_norm.T
            return similarity.squeeze()
    
    classifier = SimClassifier(trained_prompts).to(device)
    classifier.eval()
    
    print(f"Loaded Q16 NSFW classifier with {trained_prompts.shape[0]} categories")
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(img) for img in images]
        else:
            pil_images = images
        
        rewards = []
        
        with torch.no_grad():
            for pil_image in pil_images:
                # Preprocess image with CLIP
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Get image features
                image_features = clip_model.encode_image(image_input).half()
                
                # Get similarity scores
                similarities = classifier(image_features)
                prob = torch.softmax(similarities, dim=-1)

                pred_class = 1 # NSFW class index
                reward = (1 - prob[pred_class].item()) # * 10.0 # Scale to 0-10
                rewards.append(reward)

        return np.array(rewards), {}
    
    return _fn

def nsfw_clip_unlearning():
    """NSFW unlearning reward function using CLIP binary classifier
    
    Returns scores from 0 to 10, where:
    - 0-3: Very NSFW (high probability for NSFW class)
    - 4-6: Borderline (uncertain classification)
    - 7-10: Safe content (high probability for Safe class)
    
    Uses binary classification with probabilistic scoring.
    """
    import os
    from timestep_aware_clip import TimestepAwareCLIPVisionModel
    from train_critic_nsfw_classification import BinaryClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained NSFW binary classifier
    critic_path = "checkpoints/nsfw_critic_best_e35_s1945_loss_0.2656.pt"  # Update with your checkpoint
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"NSFW classifier checkpoint not found at {critic_path}")
    
    checkpoint = torch.load(critic_path, map_location=device)
    
    # Load backbone
    backbone = TimestepAwareCLIPVisionModel.from_pretrained_clip(
        "openai/clip-vit-base-patch32",
        gate_add_one=True,
        time_embed_mult=4
    ).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    
    # Load binary classifier
    feat_dim = backbone.config.hidden_size
    classifier = BinaryClassifier(in_dim=feat_dim).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    backbone.eval()
    classifier.eval()
    
    # CLIP preprocessing
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor(clip_proc.image_processor.image_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(clip_proc.image_processor.image_std, device=device).view(1, -1, 1, 1)
    
    resize_to = 224
    
    print(f"Loaded NSFW binary classifier from {critic_path}")
    
    def _fn(images, prompts, metadata, timesteps=None):
        if isinstance(images, torch.Tensor):
            # Convert from tensor to PIL images
            if images.dtype != torch.uint8:
                images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            images = images.cpu().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(img) for img in images]
        else:
            pil_images = images
        
        rewards = []
        
        with torch.no_grad():
            for pil_image in pil_images:
                try:
                    # Convert to tensor and resize
                    img_tensor = T.ToTensor()(pil_image).unsqueeze(0).to(device)
                    img_tensor = F.interpolate(
                        img_tensor,
                        size=(resize_to, resize_to),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True
                    )
                    
                    # CLIP normalization
                    pixel_values = (img_tensor - mean) / std
                    
                    # Timestep is 0 for final denoised images
                    tau = torch.zeros(1, device=device)
                    
                    # Get classification probabilities
                    out = backbone(pixel_values=pixel_values, timesteps=tau)
                    logits = classifier(out.pooler_output)  # [1, 2]
                    probs = F.softmax(logits, dim=-1)  # [1, 2]
                    
                    # Get Safe probability (class 0)
                    safe_prob = probs[0, 0].item()
                    
                    # Scale to [0, 10] range
                    # safe_prob = 1.0 -> reward = 10 (very safe)
                    # safe_prob = 0.0 -> reward = 0 (very NSFW)
                    reward = safe_prob * 10.0
                    rewards.append(reward)
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    rewards.append(5.0)  # Neutral reward on error
        
        rewards = np.array(rewards)
        return rewards, {}
    
    return _fn


class CriticNSFWClassificationWrapper:
    """Wrapper for NSFW binary classification critic with timestep awareness"""
    def __init__(self, backbone, classifier, mean, std, resize_to):
        self.backbone = backbone
        self.classifier = classifier
        self.mean = mean
        self.std = std
        self.resize_to = resize_to
    
    def __call__(self, images, timesteps, prompts, metadata):
        """
        Critic function that predicts value functions using binary classification.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width) - decoded images
            timesteps: Tensor of shape (batch_size,) - diffusion timesteps  
            prompts: List of prompts
            metadata: Additional metadata
            
        Returns:
            values: Tensor of shape (batch_size,) - predicted value functions (0-10 scale)
            metadata: Additional metadata
        """
        with torch.no_grad():
            images = images.to(dtype=torch.float32)
            if images.max() > 1.0:
                images = images / 255.0
            
            # Resize to 224x224
            images = F.interpolate(
                images, 
                size=(self.resize_to, self.resize_to),
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
            
            # CLIP normalization
            pixel_values = (images - self.mean) / self.std
            
            # Normalize timesteps to [0, 1] range
            tau = timesteps.to(images.device).float()
            if tau.max() > 1.0:
                tau = tau / 1000.0
            
            # Get classification predictions
            out = self.backbone(pixel_values=pixel_values, timesteps=tau)
            logits = self.classifier(out.pooler_output)  # [B, 2]
            probs = F.softmax(logits, dim=-1)  # [B, 2]
            
            # Get Safe probability (class 0) and scale to [0, 10]
            safe_probs = probs[:, 0]  # [B]
            values = safe_probs * 10.0  # Scale to 0-10 range
            
        return values, {}


def nsfw_clip_critic():
    """NSFW unlearning critic function using timestep-aware CLIP binary classifier for Actor-Critic training
    
    Returns value estimates from 0 to 10, where:
    - 0-3: Predicted NSFW content
    - 4-6: Uncertain/borderline
    - 7-10: Predicted Safe content
    """
    import os
    from timestep_aware_clip import TimestepAwareCLIPVisionModel
    from train_critic_nsfw_classification import BinaryClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained NSFW binary classifier
    critic_path = "checkpoints/nsfw_critic_best_e35_s1945_loss_0.2656.pt"  # Update with your checkpoint
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"NSFW classifier not found at {critic_path}")
    
    checkpoint = torch.load(critic_path, map_location=device)
    
    # Load backbone
    backbone = TimestepAwareCLIPVisionModel.from_pretrained_clip(
        "openai/clip-vit-base-patch32",
        gate_add_one=True,
        time_embed_mult=4
    ).to(device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    
    # Load binary classifier
    feat_dim = backbone.config.hidden_size
    classifier = BinaryClassifier(in_dim=feat_dim).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    backbone.eval()
    classifier.eval()
    
    # CLIP preprocessing
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor(clip_proc.image_processor.image_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(clip_proc.image_processor.image_std, device=device).view(1, -1, 1, 1)
    
    resize_to = 224
    
    print(f"Loaded NSFW binary classifier critic from {critic_path}")
    print(f"Backbone parameters: {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")

    return CriticNSFWClassificationWrapper(
        backbone, 
        classifier, 
        mean=mean, 
        std=std, 
        resize_to=resize_to
    )