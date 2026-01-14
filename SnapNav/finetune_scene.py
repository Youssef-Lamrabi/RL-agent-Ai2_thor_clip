#!/usr/bin/env python3
"""
Fast Fine-tuning Script for Scene-Specific Object Navigation

Uses behavioral cloning with expert demonstrations (shortest paths).
Training is fast because we pre-compute all demonstrations.

Usage:
    python finetune_scene.py
"""

import os
import sys
import copy
import random
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import from existing find_object.py
from find_object import (
    ObjectNavAgent, 
    ObjectNavActorCritic,
    CLIPResNetPreprocessor,
    TARGET_OBJECTS,
    ACTIONS,
)

from ai2thor.controller import Controller

# Configuration
SCENE = "FloorPlan1"
TARGET_OBJECTS_TO_TRAIN = ["Apple", "GarbageCan", "Mug"]
EPISODES_PER_OBJECT = 30  # Fast training
EPOCHS = 10  # Quick training
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  # Small LR for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(
    SCRIPT_DIR, 
    "pretrained_models", 
    "exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt"
)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "pretrained_models", "finetuned_floorplan1.pt")


class ExpertDemonstrationDataset(Dataset):
    """Dataset of expert demonstrations for behavioral cloning."""
    
    def __init__(self, demonstrations: List[Dict]):
        self.demonstrations = demonstrations
    
    def __len__(self):
        return len(self.demonstrations)
    
    def __getitem__(self, idx):
        demo = self.demonstrations[idx]
        return {
            "features": demo["features"],
            "goal_idx": demo["goal_idx"],
            "action": demo["action"],
            "prev_action": demo["prev_action"],
        }


def get_shortest_path_actions(controller: Controller, target_obj: Dict) -> List[str]:
    """Get shortest path actions to target object using AI2-THOR pathfinding."""
    target_pos = target_obj["position"]
    
    # Get path to object
    try:
        path = controller.step(
            action="GetShortestPathToPoint",
            position=target_pos,
            allowedError=0.5
        )
        
        if not path.metadata["lastActionSuccess"]:
            return []
        
        positions = path.metadata["actionReturn"]["corners"]
        if not positions or len(positions) < 2:
            return []
        
        # Convert path to actions
        actions = []
        agent_pos = controller.last_event.metadata["agent"]["position"]
        agent_rot = controller.last_event.metadata["agent"]["rotation"]["y"]
        
        for next_pos in positions[1:]:
            # Calculate direction to next point
            dx = next_pos["x"] - agent_pos["x"]
            dz = next_pos["z"] - agent_pos["z"]
            target_angle = np.degrees(np.arctan2(dx, dz)) % 360
            
            # Rotate to face target
            angle_diff = (target_angle - agent_rot + 180) % 360 - 180
            
            while abs(angle_diff) > 15:  # 30 degree rotation steps
                if angle_diff > 0:
                    actions.append("RotateRight")
                    agent_rot = (agent_rot + 30) % 360
                else:
                    actions.append("RotateLeft")
                    agent_rot = (agent_rot - 30) % 360
                angle_diff = (target_angle - agent_rot + 180) % 360 - 180
            
            # Move forward
            actions.append("MoveAhead")
            agent_pos = next_pos
        
        actions.append("End")
        return actions[:50]  # Cap at 50 actions
        
    except Exception as e:
        print(f"Path finding error: {e}")
        return []


def generate_simple_exploration_demo(controller: Controller, target_type: str) -> List[Tuple[str, np.ndarray]]:
    """Generate a simple demonstration by random exploration until object found."""
    actions_frames = []
    max_steps = 100
    
    for _ in range(max_steps):
        frame = controller.last_event.frame
        
        # Check if target is visible
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectType"] == target_type and obj["visible"]:
                actions_frames.append(("End", frame))
                return actions_frames
        
        # Simple exploration: prefer forward movement
        probs = [0.5, 0.2, 0.2, 0.0, 0.05, 0.05]  # MoveAhead, RotateLeft, RotateRight, End, LookUp, LookDown
        action = random.choices(ACTIONS, weights=probs)[0]
        
        actions_frames.append((action, frame))
        event = controller.step(action=action)
        
        if not event.metadata["lastActionSuccess"] and action == "MoveAhead":
            # Hit wall, rotate
            controller.step(action="RotateRight")
    
    return []  # Failed to find


def collect_demonstrations(
    controller: Controller,
    preprocessor: CLIPResNetPreprocessor,
    target_objects: List[str],
    episodes_per_object: int,
) -> List[Dict]:
    """Collect expert demonstrations for all target objects."""
    all_demos = []
    
    for target_type in target_objects:
        target_idx = TARGET_OBJECTS.index(target_type)
        print(f"\nCollecting demonstrations for {target_type}...")
        
        successful_episodes = 0
        attempts = 0
        
        while successful_episodes < episodes_per_object and attempts < episodes_per_object * 3:
            attempts += 1
            
            # Reset scene with random agent position
            controller.reset(scene=SCENE)
            
            # Randomize agent starting position
            positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
            if positions:
                start_pos = random.choice(positions)
                controller.step(
                    action="Teleport",
                    position=start_pos,
                    rotation={"x": 0, "y": random.randint(0, 11) * 30, "z": 0},
                    horizon=15
                )
            
            # Generate demonstration
            demo_data = generate_simple_exploration_demo(controller, target_type)
            
            if len(demo_data) < 3:
                continue  # Skip very short or failed demos
            
            # Process demonstration into training data
            prev_action = 0
            for action_name, frame in demo_data:
                action_idx = ACTIONS.index(action_name)
                
                # Extract features
                with torch.no_grad():
                    features = preprocessor.extract_features(frame)
                
                all_demos.append({
                    "features": features.cpu().squeeze(0),
                    "goal_idx": torch.tensor(target_idx),
                    "action": torch.tensor(action_idx),
                    "prev_action": torch.tensor(prev_action),
                })
                
                prev_action = action_idx
            
            successful_episodes += 1
            print(f"  Episode {successful_episodes}/{episodes_per_object} collected ({len(demo_data)} steps)")
    
    print(f"\nTotal demonstrations collected: {len(all_demos)}")
    return all_demos


def train_model(
    model: ObjectNavActorCritic,
    demonstrations: List[Dict],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> ObjectNavActorCritic:
    """Train model using behavioral cloning."""
    print("\n" + "="*50)
    print("Starting Behavioral Cloning Training")
    print("="*50)
    
    dataset = ExpertDemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Only train actor (policy) head and goal encoder
    optimizer = optim.Adam([
        {"params": model.goal_visual_encoder.parameters(), "lr": lr},
        {"params": model.actor.parameters(), "lr": lr * 2},  # Higher LR for actor
    ], lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            features = batch["features"].to(device)
            goal_idx = batch["goal_idx"].to(device)
            actions = batch["action"].to(device)
            prev_actions = batch["prev_action"].to(device)
            
            # Initialize hidden state
            hidden = torch.zeros(1, features.size(0), model.hidden_size, device=device)
            mask = torch.ones(features.size(0), device=device)
            
            # Forward pass
            action_dist, _, _ = model(features, goal_idx, hidden, prev_actions, mask)
            
            # Compute loss
            logits = action_dist.logits
            loss = criterion(logits, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            predicted = logits.argmax(dim=1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
    
    model.eval()
    return model


def main():
    print("="*60)
    print("  Fast Fine-tuning for FloorPlan1")
    print("="*60)
    print(f"\nScene: {SCENE}")
    print(f"Target objects: {TARGET_OBJECTS_TO_TRAIN}")
    print(f"Episodes per object: {EPISODES_PER_OBJECT}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print()
    
    # Load preprocessor
    print("Loading CLIP preprocessor...")
    preprocessor = CLIPResNetPreprocessor(model_type="RN50", device=DEVICE)
    
    # Load pretrained model
    print("Loading pretrained model...")
    model = ObjectNavActorCritic(
        num_objects=len(TARGET_OBJECTS),
        num_actions=len(ACTIONS),
        hidden_size=512,
        goal_dims=32,
    ).to(DEVICE)
    
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    # Initialize controller
    print("Initializing AI2-THOR...")
    controller = Controller(
        scene=SCENE,
        gridSize=0.25,
        rotateStepDegrees=30,
        visibilityDistance=1.5,
        width=224,  # Smaller for faster processing
        height=224,
        fieldOfView=90,
        agentMode="default",
        renderDepthImage=False,
        snapToGrid=False,  # Required for non-90-degree rotations
    )
    
    # Collect demonstrations
    demonstrations = collect_demonstrations(
        controller, preprocessor, TARGET_OBJECTS_TO_TRAIN, EPISODES_PER_OBJECT
    )
    
    if not demonstrations:
        print("Error: No demonstrations collected!")
        controller.stop()
        return
    
    # Train
    model = train_model(model, demonstrations, EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE)
    
    # Save
    print(f"\nSaving fine-tuned model to {OUTPUT_PATH}...")
    torch.save({"model_state_dict": model.state_dict()}, OUTPUT_PATH)
    print("Done!")
    
    controller.stop()
    
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"\nTo use the fine-tuned model, update find_object.py to load:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
