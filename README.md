# AI2-THOR Object Navigation Agent

A pretrained ProcTHOR-RL object navigation agent that can find objects in AI2-THOR environments using natural language commands.

## Overview

This project provides an inference script (`find_object.py`) that uses a pretrained ProcTHOR-RL model with CLIP visual encoding to navigate AI2-THOR environments and locate target objects. The agent can understand commands like "find the apple" and autonomously navigate to the target.

### Features

- 🤖 **Natural language object search** - Just tell the agent what to find
- 🧠 **CLIP-based visual encoding** - Uses ResNet50 features from OpenAI CLIP
- 🎯 **16 supported object types** - From common household items to furniture
- 🔄 **Stuck detection & recovery** - Intelligent exploration when agent gets stuck
- 🎮 **Interactive mode** - Real-time navigation with live feedback
- ⚡ **Fine-tuning support** - Quickly adapt to specific scenes

## Supported Objects

The agent can find the following objects:

- `AlarmClock`, `Apple`, `BaseballBat`, `BasketBall`
- `Bed`, `Bowl`, `Chair`, `GarbageCan`
- `HousePlant`, `Laptop`, `Mug`, `Sofa`
- `SprayBottle`, `Television`, `Toilet`, `Vase`

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai2thor-language-agent.git
   cd ai2thor-language-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Linux/macOS:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained models**

   You need to download at least one pretrained model checkpoint. Choose one:

   **Option A: Original ProcTHOR-RL Model (Recommended for general use)**
   ```bash
   # Create models directory
   mkdir -p pretrained_models
   
   # Download using the ProcTHOR-RL script
   python -c "import sys; sys.path.insert(0, 'procthor-rl'); from scripts.download_ckpt import main; main()" \
     --save_dir pretrained_models \
     --ckpt_ids CLIP-GRU
   ```

   The file should be named: `exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt`

   **Option B: Fine-tuned Model for FloorPlan1 (Faster for specific scene)**
   
   If you have the fine-tuned model (`finetuned_floorplan1.pt`), place it in the `pretrained_models/` directory:
   ```bash
   # Place your fine-tuned model here:
   # pretrained_models/finetuned_floorplan1.pt
   ```

   > **Note**: The fine-tuned model is optimized for FloorPlan1 and provides faster, more reliable navigation for that specific scene.

## Usage

### Basic Usage

Run the interactive agent:

```bash
python find_object.py
```

The agent will start in FloorPlan1. You can give commands like:

```
>>> find the apple
>>> find the mug
>>> find the garbage can
```

### Commands

- `find <object>` - Search for the specified object
- `scene <name>` - Switch to a different scene (e.g., `scene FloorPlan2`)
- `reset` - Reset the current scene
- `quit` or `exit` - Exit the program

### Example Session

```bash
$ python find_object.py

==============================================================
  ProcTHOR-RL Object Navigation Agent
==============================================================

Using checkpoint: finetuned_floorplan1.pt
Using device: cuda
Loading CLIP ResNet50...
Building model...
Loading checkpoint from pretrained_models/finetuned_floorplan1.pt...
Loaded 20 parameters successfully!
Model ready!

Supported objects:
AlarmClock, Apple, BaseballBat, BasketBall, Bed, Bowl, Chair, GarbageCan, 
HousePlant, Laptop, Mug, Sofa, SprayBottle, Television, Toilet, Vase

Commands: 'find <object>', 'scene <name>', 'reset', 'quit'

>>> find the apple
Searching for: Apple
----------------------------------------
✓ Found Apple at distance 1.32m after 47 steps!
----------------------------------------
SUCCESS: Found Apple in 47 steps
```

## Fine-tuning (Optional)

To fine-tune the model for faster, more reliable navigation on specific scenes:

```bash
python finetune_scene.py
```

This script:
1. Collects expert demonstrations using random exploration
2. Trains the model using behavioral cloning
3. Saves the fine-tuned model to `pretrained_models/finetuned_floorplan1.pt`

### Fine-tuning Configuration

Edit `finetune_scene.py` to customize:

- `SCENE` - Target scene (default: "FloorPlan1")
- `TARGET_OBJECTS_TO_TRAIN` - Objects to train on (default: ["Apple", "GarbageCan", "Mug"])
- `EPISODES_PER_OBJECT` - Training episodes per object (default: 30)
- `EPOCHS` - Training epochs (default: 10)

## Project Structure

```
ai2thor-language-agent/
├── find_object.py           # Main inference script ⭐
├── finetune_scene.py        # Fine-tuning script (optional)
├── pretrained_models/       # Model checkpoints (gitignored)
│   ├── exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt
│   └── finetuned_floorplan1.pt
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore configuration
└── README.md               # This file
```

## Technical Details

### Architecture

- **Visual Encoder**: CLIP ResNet50 pretrained on image-text pairs
- **Goal Encoder**: Learned embedding for each object type
- **State Encoder**: GRU (Gated Recurrent Unit) for temporal reasoning
- **Policy**: Actor-Critic architecture trained with DD-PPO

### Navigation Strategy

The agent uses several techniques to improve navigation:

1. **Confidence-based action selection** - Falls back to exploration when uncertain
2. **Stuck pattern detection** - Detects repetitive actions (e.g., rotating back and forth)
3. **Exploration mode** - Switches to systematic exploration when stuck
4. **Proactive object checking** - Auto-terminates when target is visible and close
5. **Collision recovery** - Rotates when blocked instead of repeatedly failing

### Model Checkpoints

The model checkpoint priority (from highest to lowest):

1. `finetuned_floorplan1.pt` - Fine-tuned for FloorPlan1 (13MB)
2. `exp_ObjectNav-RGB-ClipResNet50GRU-DDPPO__stage_02__steps_000415481616.pt` - Pretrained (40MB)

## Troubleshooting

### "No checkpoint found" Error

Make sure you've downloaded a model checkpoint to `pretrained_models/`. See the [Installation](#installation) section.

### CUDA Out of Memory

If you encounter GPU memory errors, the model will automatically fall back to CPU. You can also explicitly use CPU:

```python
# In find_object.py, line 264
agent = ObjectNavAgent(checkpoint_path, device="cpu")
```

### Agent Getting Stuck

The agent has built-in stuck detection, but if navigation is poor:

1. Try the fine-tuned model for better scene-specific performance
2. Adjust parameters in `find_object.py`:
   - `STUCK_THRESHOLD` (line 256) - Lower to detect stuck faster
   - `CONFIDENCE_THRESHOLD` (line 257) - Adjust confidence threshold

### AI2-THOR Installation Issues

If AI2-THOR fails to install or run:

```bash
# Install system dependencies (Linux)
sudo apt-get update
sudo apt-get install xorg libxcb-xinerama0

# For headless servers, use xvfb
pip install xvfb-run
xvfb-run python find_object.py
```

## Performance

### Navigation Success Rates

On FloorPlan1 with fine-tuned model:

| Object | Success Rate | Avg Steps |
|--------|-------------|-----------|
| Apple | ~85% | 52 |
| GarbageCan | ~80% | 61 |
| Mug | ~75% | 58 |

*Note: Performance varies by starting position and scene layout*

## Dependencies

- **PyTorch** (>=1.9.0) - Deep learning framework
- **CLIP** - OpenAI's vision-language model
- **AI2-THOR** - Interactive 3D environment
- **NumPy** - Numerical computing
- **Pillow** - Image processing

See `requirements.txt` for complete list.

## Citation

This project uses the ProcTHOR-RL model. If you use this work, please cite:

```bibtex
@inproceedings{procthor-rl,
  title={ProcTHOR: Large-Scale Embodied AI Using Procedural Generation},
  author={Deitke, Matt and others},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

This project is for educational and research purposes. The pretrained models are subject to the [AllenAI License](https://allenai.org/licenses).

## Acknowledgments

- [AllenAI](https://allenai.org/) for AI2-THOR and ProcTHOR
- [OpenAI](https://openai.com/) for CLIP
- ProcTHOR-RL team for the pretrained models

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Object Navigation! 🤖🎯**
