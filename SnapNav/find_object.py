#!/usr/bin/env python3
"""
ProcTHOR-RL Object Navigation Agent

This script allows you to navigate AI2-THOR environments using a pretrained
ProcTHOR-RL model. You can give natural language commands like "find the apple"
and the agent will navigate to the target object.

Usage:
    python find_object.py

Supported objects:
    AlarmClock, Apple, BaseballBat, BasketBall, Bed, Bowl, Chair, GarbageCan,
    HousePlant, Laptop, Mug, Sofa, SprayBottle, Television, Toilet, Vase
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add procthor-rl to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCTHOR_DIR = os.path.join(SCRIPT_DIR, "procthor-rl")
sys.path.insert(0, PROCTHOR_DIR)

from ai2thor.controller import Controller

# CLIP for visual encoding
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    os.system("pip install git+https://github.com/openai/CLIP.git")
    import clip

# Supported target objects (must match training order)
TARGET_OBJECTS = [
    "AlarmClock", "Apple", "BaseballBat", "BasketBall", "Bed", "Bowl",
    "Chair", "GarbageCan", "HousePlant", "Laptop", "Mug", "Sofa",
    "SprayBottle", "Television", "Toilet", "Vase"
]

# Available actions
ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "End", "LookUp", "LookDown"]

# CLIP normalization constants
RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
RGB_STDS = (0.26862954, 0.26130258, 0.27577711)


class CLIPResNetPreprocessor:
    """Preprocessor that extracts CLIP ResNet features from RGB images."""
    
    def __init__(self, model_type: str = "RN50", device: str = "cuda"):
        self.device = device
        self.model, _ = clip.load(model_type, device=device)
        self.model.eval()
        self.visual_model = self.model.visual
        
    @torch.no_grad()
    def preprocess_frame(self, frame: np.ndarray, size: int = 224) -> torch.Tensor:
        """Preprocess a frame for CLIP."""
        image = Image.fromarray(frame)
        image = image.resize((size, size), Image.BILINEAR)
        
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        for i in range(3):
            tensor[i] = (tensor[i] - RGB_MEANS[i]) / RGB_STDS[i]
            
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def extract_features(self, frame: np.ndarray) -> torch.Tensor:
        """Extract CLIP ResNet features from frame."""
        x = self.preprocess_frame(frame)
        
        def stem(x):
            x = self.visual_model.relu1(self.visual_model.bn1(self.visual_model.conv1(x)))
            x = self.visual_model.relu2(self.visual_model.bn2(self.visual_model.conv2(x)))
            x = self.visual_model.relu3(self.visual_model.bn3(self.visual_model.conv3(x)))
            x = self.visual_model.avgpool(x)
            return x
        
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_model.layer1(x)
        x = self.visual_model.layer2(x)
        x = self.visual_model.layer3(x)
        x = self.visual_model.layer4(x)
        
        return x.float()


class ResnetTensorGoalEncoder(nn.Module):
    """Goal-conditioned visual encoder matching ProcTHOR-RL architecture."""
    
    def __init__(
        self,
        num_objects: int = 16,
        resnet_channels: int = 2048,
        goal_embed_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):
        super().__init__()
        
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        
        # Goal embedding
        self.embed_goal = nn.Embedding(num_objects, goal_embed_dims)
        
        # ResNet compressor
        self.resnet_compressor = nn.Sequential(
            nn.Conv2d(resnet_channels, resnet_compressor_hidden_out_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(*resnet_compressor_hidden_out_dims, 1),
            nn.ReLU(),
        )
        
        # Target-observation combiner (32 + 32 = 64 input channels)
        self.target_obs_combiner = nn.Sequential(
            nn.Conv2d(
                resnet_compressor_hidden_out_dims[1] + goal_embed_dims,
                combiner_hidden_out_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(*combiner_hidden_out_dims, 1),
        )
        
        self.spatial_size = 7  # For RN50 with 224x224 input
        
    @property
    def output_dims(self):
        return self.combine_hid_out_dims[-1] * self.spatial_size * self.spatial_size
    
    def forward(self, resnet_features: torch.Tensor, goal_idx: torch.Tensor) -> torch.Tensor:
        B = resnet_features.shape[0]
        H, W = resnet_features.shape[2], resnet_features.shape[3]
        
        # Compress ResNet features
        obs_embs = self.resnet_compressor(resnet_features)
        
        # Get goal embedding and expand spatially
        goal_emb = self.embed_goal(goal_idx)
        goal_emb = goal_emb.view(B, self.goal_embed_dims, 1, 1).expand(-1, -1, H, W)
        
        # Combine
        combined = torch.cat([obs_embs, goal_emb], dim=1)
        x = self.target_obs_combiner(combined)
        x = x.reshape(B, -1)
        
        return x


class ObjectNavActorCritic(nn.Module):
    """Actor-Critic model matching ProcTHOR-RL CLIP-GRU architecture."""
    
    def __init__(
        self,
        num_objects: int = 16,
        num_actions: int = 6,
        hidden_size: int = 512,
        goal_dims: int = 32,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # Visual encoder
        self.goal_visual_encoder = ResnetTensorGoalEncoder(
            num_objects=num_objects,
            goal_embed_dims=goal_dims,
        )
        
        # Previous action embedding (Linear layer as in checkpoint)
        self.prev_action_embedder = nn.ModuleDict({
            'fc': nn.Linear(num_actions, num_actions)
        })
        
        # RNN input size = 1568 (visual) + 6 (action) = 1574
        rnn_input_size = self.goal_visual_encoder.output_dims + num_actions
            
        # State encoder (GRU) - matching checkpoint structure
        self.state_encoders = nn.ModuleDict({
            'single_belief': nn.ModuleDict({
                'rnn': nn.GRU(
                    input_size=rnn_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=False,
                )
            })
        })
        
        # Actor head (matching checkpoint: linear layer)
        self.actor = nn.ModuleDict({
            'linear': nn.Linear(hidden_size, num_actions)
        })
        
        # Critic head
        self.critic = nn.ModuleDict({
            'fc': nn.Linear(hidden_size, 1)
        })
        
    def forward(
        self,
        resnet_features: torch.Tensor,
        goal_idx: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Encode observation with goal
        obs_embed = self.goal_visual_encoder(resnet_features, goal_idx)
        
        # One-hot encode previous action and pass through fc
        action_onehot = torch.zeros(prev_action.shape[0], self.num_actions, device=prev_action.device)
        action_onehot.scatter_(1, prev_action.unsqueeze(1), 1.0)
        action_embed = self.prev_action_embedder['fc'](action_onehot)
        
        # Combine embeddings
        obs_embed = torch.cat([obs_embed, action_embed], dim=-1)
        obs_embed = obs_embed.unsqueeze(0)  # (1, B, input_size)
        
        # Apply mask to hidden state
        hidden_state = hidden_state * mask.view(1, -1, 1)
        
        # RNN forward
        rnn_out, new_hidden = self.state_encoders['single_belief']['rnn'](obs_embed, hidden_state)
        beliefs = rnn_out.squeeze(0)
        
        # Actor-Critic heads
        logits = self.actor['linear'](beliefs)
        action_dist = torch.distributions.Categorical(logits=logits)
        value = self.critic['fc'](beliefs)
        
        return action_dist, value, new_hidden


class ObjectNavAgent:
    """Agent for object navigation using pretrained ProcTHOR-RL model."""
    
    # Constants for improved navigation
    STUCK_THRESHOLD = 6  # Number of repeated actions to consider stuck
    CONFIDENCE_THRESHOLD = 0.35  # Minimum confidence to trust model action
    VISIBILITY_END_THRESHOLD = 1.5  # Distance threshold to auto-end when object visible
    EXPLORATION_AFTER_STUCK = 5  # Random exploration steps after getting stuck
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        print(f"Using device: {device}")
        
        # Load CLIP preprocessor
        print("Loading CLIP ResNet50...")
        self.preprocessor = CLIPResNetPreprocessor(model_type="RN50", device=device)
        
        # Build model
        print("Building model...")
        self.model = ObjectNavActorCritic(
            num_objects=len(TARGET_OBJECTS),
            num_actions=len(ACTIONS),
            hidden_size=512,
            goal_dims=32,
        ).to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self._load_checkpoint(checkpoint_path)
        
        self.model.eval()
        print("Model ready!")
        
    def _load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Load directly - our model structure matches the checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"Note: Missing keys (may be auxiliary modules): {missing}")
        if unexpected:
            print(f"Note: Unexpected keys: {unexpected}")
        
        print(f"Loaded {len(state_dict) - len(unexpected)} parameters successfully!")
    
    def parse_command(self, command: str) -> Optional[str]:
        """Extract target object from natural language command."""
        command_lower = command.lower().strip()
        
        # Direct object name matching
        for obj in TARGET_OBJECTS:
            if obj.lower() in command_lower:
                return obj
        
        # Common variations
        variations = {
            "tv": "Television",
            "plant": "HousePlant",
            "house plant": "HousePlant",
            "garbage": "GarbageCan",
            "trash": "GarbageCan",
            "trash can": "GarbageCan",
            "baseball": "BaseballBat",
            "bat": "BaseballBat",
            "basketball": "BasketBall",
            "ball": "BasketBall",
            "spray": "SprayBottle",
            "bottle": "SprayBottle",
            "clock": "AlarmClock",
            "alarm": "AlarmClock",
            "cup": "Mug",
            "couch": "Sofa",
            "computer": "Laptop",
        }
        
        for variation, obj in variations.items():
            if variation in command_lower:
                return obj
        
        return None
    
    def _check_object_visible(self, controller: Controller, target_object: str) -> Tuple[bool, Optional[float]]:
        """Check if target object is visible and return its distance."""
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectType"] == target_object and obj["visible"]:
                distance = obj.get("distance", None)
                return True, distance
        return False, None
    
    def _check_object_exists(self, controller: Controller, target_object: str) -> bool:
        """Check if target object exists in the current scene."""
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectType"] == target_object:
                return True
        return False
    
    def _detect_stuck_pattern(self, actions: list) -> bool:
        """Detect if agent is stuck in a repetitive action pattern."""
        if len(actions) < self.STUCK_THRESHOLD:
            return False
        
        recent = actions[-self.STUCK_THRESHOLD:]
        
        # Check for alternating pattern (e.g., RotateLeft, RotateRight, RotateLeft...)
        if len(set(recent)) <= 2 and "MoveAhead" not in recent:
            return True
        
        # Check for same action repeated (not MoveAhead which is ok)
        if len(set(recent)) == 1 and recent[0] != "MoveAhead":
            return True
            
        return False
    
    def _get_exploration_action(self, step: int, last_action: str) -> str:
        """Get an exploration action to escape stuck states."""
        # Simple exploration: try moving forward, if blocked rotate
        exploration_sequence = ["MoveAhead", "MoveAhead", "RotateRight", "MoveAhead", "MoveAhead", "RotateLeft"]
        return exploration_sequence[step % len(exploration_sequence)]
    
    @torch.no_grad()
    def find_object(
        self,
        target_object: str,
        controller: Controller,
        max_steps: int = 500,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Navigate to find the target object with improved stuck detection and exploration."""
        if target_object not in TARGET_OBJECTS:
            raise ValueError(f"Unknown object: {target_object}. Valid: {TARGET_OBJECTS}")
        
        # Check if object exists in scene
        if not self._check_object_exists(controller, target_object):
            if verbose:
                print(f"⚠ Warning: {target_object} may not exist in this scene!")
        
        goal_idx = torch.tensor([TARGET_OBJECTS.index(target_object)], device=self.device)
        
        # Initialize hidden state
        hidden = torch.zeros(1, 1, self.model.hidden_size, device=self.device)
        prev_action = torch.tensor([0], device=self.device)
        mask = torch.tensor([0.0], device=self.device)
        
        success = False
        frames = []
        actions_taken = []
        exploration_mode = False
        exploration_steps = 0
        stuck_count = 0
        
        for step in range(max_steps):
            frame = controller.last_event.frame
            frames.append(frame)
            
            # Proactive check: Is target object already visible?
            is_visible, distance = self._check_object_visible(controller, target_object)
            if is_visible and distance is not None and distance < self.VISIBILITY_END_THRESHOLD:
                success = True
                if verbose:
                    print(f"✓ Found {target_object} at distance {distance:.2f}m after {step+1} steps!")
                break
            
            # Check for stuck pattern
            if self._detect_stuck_pattern(actions_taken):
                if not exploration_mode:
                    exploration_mode = True
                    exploration_steps = 0
                    stuck_count += 1
                    if verbose:
                        print(f"  [Step {step}] Stuck detected! Entering exploration mode (attempt {stuck_count})")
            
            # Determine action
            if exploration_mode and exploration_steps < self.EXPLORATION_AFTER_STUCK:
                # Use exploration action to escape
                action = self._get_exploration_action(exploration_steps, actions_taken[-1] if actions_taken else "")
                action_idx = ACTIONS.index(action)
                exploration_steps += 1
                if exploration_steps >= self.EXPLORATION_AFTER_STUCK:
                    exploration_mode = False
                    if verbose:
                        print(f"  [Step {step}] Exploration complete, resuming model-guided navigation")
            else:
                # Extract visual features
                resnet_features = self.preprocessor.extract_features(frame)
                
                # Get action from model
                action_dist, value, hidden = self.model(
                    resnet_features, goal_idx, hidden, prev_action, mask
                )
                
                probs = action_dist.probs.cpu().numpy()[0]
                max_prob = probs.max()
                action_idx = probs.argmax()
                
                # Confidence-based action selection
                if max_prob < self.CONFIDENCE_THRESHOLD:
                    # Low confidence: prefer exploration (MoveAhead) over rotation
                    if probs[0] > 0.1:  # MoveAhead has some probability
                        action_idx = 0  # MoveAhead
                    elif verbose:
                        print(f"  [Step {step}] Low confidence ({max_prob:.2f}), using model's best guess")
                
                action = ACTIONS[action_idx]
                
                if verbose and step % 20 == 0:
                    print(f"Step {step}: {action} (value: {value.item():.2f}, conf: {max_prob:.2f}, probs: {probs.round(2)})")
            
            actions_taken.append(action)
            
            # Check for End action
            if action == "End":
                is_visible, distance = self._check_object_visible(controller, target_object)
                success = is_visible
                if verbose:
                    if success:
                        print(f"✓ Found {target_object} after {step+1} steps!")
                    else:
                        print(f"✗ Called End but {target_object} not visible")
                break
            
            # Execute action
            event = controller.step(action=action)
            
            # If action failed (e.g., bumped into wall), try to recover
            if not event.metadata["lastActionSuccess"] and action == "MoveAhead":
                if verbose and step % 10 == 0:
                    print(f"  [Step {step}] MoveAhead blocked, rotating...")
                # Rotate to find new path
                controller.step(action="RotateRight")
            
            # Update for next step
            prev_action = torch.tensor([action_idx], device=self.device)
            mask = torch.tensor([1.0], device=self.device)
        
        else:
            # Check one more time if visible at end
            is_visible, _ = self._check_object_visible(controller, target_object)
            if is_visible:
                success = True
                if verbose:
                    print(f"✓ Found {target_object} (visible at end) after {max_steps} steps!")
            elif verbose:
                print(f"✗ Reached max steps ({max_steps}) without finding {target_object}")
        
        return {
            "success": success,
            "steps": step + 1 if 'step' in dir() else max_steps,
            "target": target_object,
            "final_frame": frames[-1] if frames else None,
            "actions": actions_taken,
            "stuck_recoveries": stuck_count,
        }


def create_ithor_controller(scene: str = "FloorPlan1") -> Controller:
    """Create an AI2-THOR controller with iTHOR scene."""
    print(f"Initializing AI2-THOR with scene: {scene}...")
    
    controller = Controller(
        scene=scene,
        gridSize=0.25,
        rotateStepDegrees=30,
        visibilityDistance=1.5,  # Increased for better object detection
        width=400,
        height=300,
        fieldOfView=90,
        agentMode="default",  # Taller agent with higher camera (can see tables)
        renderDepthImage=False,
        snapToGrid=False,  # Required for non-90-degree rotations
    )
    
    # Set camera to a good viewing angle (slight downward tilt to see tables)
    controller.step(action="Teleport", horizon=15)  # 15 degrees down
    
    return controller


def main():
    """Main interactive loop."""
    print("=" * 60)
    print("  ProcTHOR-RL Object Navigation Agent")
    print("=" * 60)
    print()
    
    # Find checkpoint (prefer fine-tuned, then pretrained)
    checkpoint_names = [
        "finetuned_floorplan1.pt",  # Fine-tuned for FloorPlan1 (fast!)
        "exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt",
        "exp_OnePhaseRGBClipResNet50Dagger_40proc__stage_00__steps_000065083050.pt",
    ]
    
    checkpoint_path = None
    # Check directly in pretrained_models dir
    for name in checkpoint_names:
        path = os.path.join(SCRIPT_DIR, "pretrained_models", name)
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if checkpoint_path is None:
        print("Error: No checkpoint found!")
        print("Run: python procthor-rl/scripts/download_ckpt.py --save_dir pretrained_models --ckpt_ids CLIP-GRU")
        return
    
    print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Initialize agent
    agent = ObjectNavAgent(checkpoint_path)
    
    # Initialize controller
    controller = create_ithor_controller("FloorPlan1")
    
    print()
    print("Supported objects:")
    print(", ".join(TARGET_OBJECTS))
    print()
    print("Commands: 'find <object>', 'scene <name>', 'reset', 'quit'")
    print()
    
    while True:
        try:
            command = input(">>> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if command.lower() == "reset":
                controller.reset()
                print("Scene reset!")
                continue
            
            if command.lower().startswith("scene "):
                scene_name = command[6:].strip()
                controller.reset(scene=scene_name)
                print(f"Switched to scene: {scene_name}")
                continue
            
            # Parse target object
            target = agent.parse_command(command)
            
            if target is None:
                print(f"Could not understand. Supported objects:")
                print(", ".join(TARGET_OBJECTS))
                continue
            
            print(f"Searching for: {target}")
            print("-" * 40)
            
            # Reset episode and set proper camera angle
            controller.reset()
            controller.step(action="Teleport", horizon=0)  # Reset camera to level
            
            # Run navigation
            result = agent.find_object(target, controller, max_steps=200, verbose=True)
            
            print("-" * 40)
            if result["success"]:
                print(f"SUCCESS: Found {target} in {result['steps']} steps")
            else:
                print(f"FAILED: Could not find {target} in {result['steps']} steps")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    controller.stop()


if __name__ == "__main__":
    main()
