#!/usr/bin/env python3
"""
FastAPI Backend Server for AI2-THOR Object Navigation Agent
Provides WebSocket API for real-time agent navigation streaming
"""

import os
import sys
import asyncio
import base64
import io
import json
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import from find_object.py
from find_object import (
    ObjectNavAgent,
    create_ithor_controller,
    TARGET_OBJECTS,
    ACTIONS,
)

# Global agent and controller instances
agent: Optional[ObjectNavAgent] = None
controller = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global agent, controller
    
    print("🚀 Initializing AI2-THOR agent...")
    
    # Find checkpoint
    checkpoint_names = [
        "pretrained_models/finetuned_floorplan1.pt",
        "pretrained_models/exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt",
    ]
    
    checkpoint_path = None
    for name in checkpoint_names:
        if os.path.exists(name):
            checkpoint_path = name
            break
    
    if checkpoint_path is None:
        raise RuntimeError("No checkpoint found! Place model in pretrained_models/")
    
    # Initialize agent
    agent = ObjectNavAgent(checkpoint_path)
    
    # Initialize controller with visible window for demo
    from ai2thor.controller import Controller
    import platform
    
    # Platform-specific settings for optimal Unity window visibility
    platform_config = {}
    if platform.system() == "Windows":
        platform_config["platform"] = "CloudRendering"
    
    controller = Controller(
        scene="FloorPlan1",
        gridSize=0.25,
        rotateStepDegrees=30,
        visibilityDistance=1.5,
        width=800,  # Optimized for faster frame processing
        height=600,
        fieldOfView=90,
        agentMode="default",
        renderDepthImage=False,
        snapToGrid=False,
        headless=False,  # Unity window visible from start
        **platform_config
    )
    controller.step(action="Teleport", horizon=15)
    
    print("🎮 Unity window should now be visible!")
    
    print("✅ Agent ready!")
    
    yield
    
    # Cleanup
    if controller:
        controller.stop()
    print("🛑 Server shutdown")


app = FastAPI(title="SnapNav API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert numpy frame to base64 encoded JPEG"""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "agent": "AI2-THOR Object Navigation",
        "supported_objects": TARGET_OBJECTS,
    }


@app.get("/objects")
async def get_supported_objects():
    """Get list of supported objects"""
    return {"objects": TARGET_OBJECTS}


@app.get("/parse_command/{command}")
async def parse_command(command: str):
    """Parse natural language command to extract target object"""
    target = agent.parse_command(command)
    if target:
        return {"success": True, "target": target, "command": command}
    else:
        return {
            "success": False, 
            "error": "Could not identify target object",
            "supported_objects": TARGET_OBJECTS
        }


@app.get("/preview_scene/{scene}")
async def preview_scene(scene: str):
    """Load a scene in Unity window for preview"""
    try:
        controller.reset(scene=scene)
        controller.step(action="Teleport", horizon=15)
        return {
            "success": True,
            "scene": scene,
            "message": f"Scene {scene} loaded in Unity window"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.websocket("/ws/live_stream")
async def websocket_live_stream(websocket: WebSocket):
    """
    WebSocket endpoint for continuous live scene streaming
    Provides real-time view of Unity scene at ~30 FPS
    """
    await websocket.accept()
    
    try:
        current_scene = "FloorPlan1"
        
        await websocket.send_json({
            "type": "status",
            "message": f"Live stream started - {current_scene}"
        })
        
        while True:
            try:
                # Check if client sent a message (scene change request)
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.01  # Non-blocking check
                )
                data = json.loads(message) if message else {}
                
                # Handle scene change
                if data.get("action") == "change_scene":
                    new_scene = data.get("scene")
                    if new_scene:
                        controller.reset(scene=new_scene)
                        controller.step(action="Teleport", horizon=15)
                        current_scene = new_scene
                        await websocket.send_json({
                            "type": "status",
                            "message": f"Scene changed to {new_scene}"
                        })
            except asyncio.TimeoutError:
                # No message received, continue streaming
                pass
            
            # Get current frame
            frame = controller.last_event.frame
            
            # Send frame
            await websocket.send_json({
                "type": "frame",
                "data": frame_to_base64(frame),
                "scene": current_scene
            })
            
            # ~30 FPS
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        print("Live stream client disconnected")
    except Exception as e:
        print(f"Live stream error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@app.websocket("/ws/navigate/{target_object}/{scene}")
async def websocket_navigate(websocket: WebSocket, target_object: str, scene: str = "FloorPlan1"):
    """
    WebSocket endpoint for real-time navigation
    
    Streams:
    - Video frames (base64 encoded)
    - Navigation steps
    - Agent actions
    - Success/failure status
    
    Args:
        target_object: The object to find (e.g., "Apple", "Mug")
        scene: The AI2-THOR scene to use (e.g., "FloorPlan1", "FloorPlan2", etc.)
    """
    await websocket.accept()
    
    try:
        # Validate target object
        if target_object not in TARGET_OBJECTS:
            await websocket.send_json({
                "type": "error",
                "message": f"Invalid object. Supported: {', '.join(TARGET_OBJECTS)}"
            })
            await websocket.close()
            return
        
        await websocket.send_json({
            "type": "status",
            "message": f"Loading {scene} scene and navigating to: {target_object}",
            "step": "initializing"
        })
        
        # Reset environment with specified scene
        controller.reset(scene=scene)
        controller.step(action="Teleport", horizon=0)
        
        # Send initial frame
        initial_frame = controller.last_event.frame
        await websocket.send_json({
            "type": "frame",
            "data": frame_to_base64(initial_frame),
            "step": "initial"
        })
        
        # Initialize agent state
        import torch
        goal_idx = torch.tensor([TARGET_OBJECTS.index(target_object)], device=agent.device)
        hidden = torch.zeros(1, 1, agent.model.hidden_size, device=agent.device)
        prev_action = torch.tensor([0], device=agent.device)
        mask = torch.tensor([0.0], device=agent.device)
        
        max_steps = 200
        success = False
        
        await websocket.send_json({
            "type": "status",
            "message": "Scanning environment...",
            "step": "scanning"
        })
        
        for step in range(max_steps):
            frame = controller.last_event.frame
            
            # Check if object is visible
            is_visible = False
            distance = None
            for obj in controller.last_event.metadata["objects"]:
                if obj["objectType"] == target_object and obj["visible"]:
                    is_visible = True
                    distance = obj.get("distance", None)
                    break
            
            if is_visible and distance is not None and distance < 1.5:
                success = True
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_to_base64(frame),
                    "step": step
                })
                await websocket.send_json({
                    "type": "success",
                    "message": f"Found {target_object}!",
                    "steps": step + 1,
                    "distance": distance
                })
                break
            
            # Get action from model
            with torch.no_grad():
                resnet_features = agent.preprocessor.extract_features(frame)
                action_dist, value, hidden = agent.model(
                    resnet_features, goal_idx, hidden, prev_action, mask
                )
                
                probs = action_dist.probs.cpu().numpy()[0]
                action_idx = probs.argmax()
                action = ACTIONS[action_idx]
            
            # Send frame every step for real-time visualization
            await websocket.send_json({
                "type": "frame",
                "data": frame_to_base64(frame),
                "step": step,
                "action": action,
                "confidence": float(probs[action_idx])
            })
            
            # Send action update
            await websocket.send_json({
                "type": "action",
                "action": action,
                "step": step,
                "confidence": float(probs[action_idx]),
                "value": float(value.item())
            })
            
            # Execute action
            if action == "End":
                if is_visible:
                    success = True
                    await websocket.send_json({
                        "type": "success",
                        "message": f"Found {target_object}!",
                        "steps": step + 1
                    })
                else:
                    await websocket.send_json({
                        "type": "failure",
                        "message": f"Agent called End but {target_object} not visible",
                        "steps": step + 1
                    })
                break
            
            event = controller.step(action=action)
            
            # Handle failed actions
            if not event.metadata["lastActionSuccess"] and action == "MoveAhead":
                controller.step(action="RotateRight")
            
            # Update for next step
            prev_action = torch.tensor([action_idx], device=agent.device)
            mask = torch.tensor([1.0], device=agent.device)
            
            # Faster updates for real-time streaming (~30 FPS)
            await asyncio.sleep(0.03)
        
        else:
            # Max steps reached
            is_visible = False
            for obj in controller.last_event.metadata["objects"]:
                if obj["objectType"] == target_object and obj["visible"]:
                    is_visible = True
                    success = True
                    break
            
            if success:
                await websocket.send_json({
                    "type": "success",
                    "message": f"Found {target_object} (visible at end)!",
                    "steps": max_steps
                })
            else:
                await websocket.send_json({
                    "type": "failure",
                    "message": f"Could not find {target_object}",
                    "steps": max_steps
                })
        
        # Send final frame
        final_frame = controller.last_event.frame
        await websocket.send_json({
            "type": "frame",
            "data": frame_to_base64(final_frame),
            "step": "final"
        })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("  SnapNav Backend Server")
    print("=" * 60)
    print()
    print("Starting server on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/navigate/{object}/{scene}")
    print()
    
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
