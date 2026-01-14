# SnapNav - AI2-THOR Object Navigation Demo

A stunning web application showcasing real-time AI-powered object navigation using the ProcTHOR-RL model in AI2-THOR environments.

![Live Demo](https://img.shields.io/badge/demo-live-brightgreen) ![React](https://img.shields.io/badge/React-18-blue) ![TypeScript](https://img.shields.io/badge/TypeScript-5-blue) ![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-orange)

## 🌟 Features

### Real-time Visualization
- **Persistent Live Stream**: Unity scene streamed at 30 FPS from page load
- **No Loading Screens**: Instant video feed when you open the demo
- **Scene Preview**: See the environment before starting navigation
- **Smooth Performance**: Optimized frame delivery for fluid visualization

### AI Navigation
- **Natural Language Commands**: Type commands like "find the apple" or "go to the mug"
- **Quick Select**: One-click buttons for common objects (Apple, Mug, Chair, Television)
- **Smart Agent**: ProcTHOR-RL model trained on 10,000+ procedurally generated scenes
- **Visual Feedback**: HUD overlay showing action, confidence, and progress

### Scene Selection
- **30 Floor Plans**: Choose from FloorPlan1 through FloorPlan30
- **Instant Switching**: Scenes change immediately in the live stream
- **Diverse Environments**: Kitchens, living rooms, bedrooms, bathrooms

### Modern UI/UX
- **Glassmorphic Design**: Premium modern aesthetic with subtle animations
- **Dark Mode**: Easy on the eyes with vibrant accent colors
- **Responsive Layout**: Works beautifully on desktop and mobile
- **Terminal-Style Interface**: Cyberpunk-inspired command interface

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend server running (SnapNav - see [Backend Setup](#backend-setup))

### Frontend Setup

```bash
# Clone the repository
git clone https://github.com/azizlg/navi-quest-bot.git
cd navi-quest-bot

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Backend Setup

The frontend requires the SnapNav backend to be running:

```bash
# Clone backend repository
git clone https://github.com/azizlg/SnapNav.git
cd SnapNav

# Install Python dependencies
pip install -r requirements.txt

# Download pretrained model (first time only)
# Place model in pretrained_models/ directory

# Start backend server
python backend_server.py
```

Backend will run on `http://localhost:8000`

## 🎮 How to Use

1. **Open the Demo**: Navigate to the demo section on the website
2. **Wait for Connection**: The live stream connects automatically (green indicator)
3. **Select a Scene**: Choose from 30 different FloorPlans using the dropdown
4. **Enter Command**: Type a natural language command or click a quick-select button
5. **Watch Navigation**: The AI agent navigates to find the object in real-time
6. **See Results**: Success message displays when object is found

### Supported Objects
AlarmClock, Apple, BaseballBat, BasketBall, Bed, Bowl, Chair, GarbageCan, HousePlant, Laptop, Mug, Sofa, SprayBottle, Television, Toilet, Vase

### Example Commands
- "find the apple"
- "go to the mug"
- "navigate to the chair"
- "locate the television"

## 🏗️ Architecture

### Frontend (This Repository)
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + Custom CSS
- **UI Components**: shadcn/ui
- **Animations**: Framer Motion
- **Icons**: Lucide React

### Backend Communication
- **Live Stream**: WebSocket connection to `/ws/live_stream` (30 FPS)
- **Navigation**: WebSocket connection to `/ws/navigate/{object}/{scene}`
- **Scene Control**: HTTP POST to `/preview_scene/{scene}`

### Data Flow
```
User Action → Frontend → WebSocket → Backend → AI2-THOR → Unity
                ↑                                              ↓
                └──────────── Video Frames ────────────────────┘
```

## 📂 Project Structure

```
navi-quest-bot/
├── src/
│   ├── components/
│   │   ├── Demo.tsx           # Main demo component
│   │   ├── Hero.tsx           # Landing page hero
│   │   ├── Features.tsx       # Features section
│   │   └── ui/                # shadcn/ui components
│   ├── pages/
│   │   └── Index.tsx          # Main page
│   ├── index.css              # Global styles
│   └── main.tsx               # App entry point
├── public/                    # Static assets
├── index.html                 # HTML template
├── package.json               # Dependencies
├── tailwind.config.ts         # Tailwind config
└── vite.config.ts            # Vite config
```

## 🛠️ Development

### Running Locally
```bash
npm run dev      # Start dev server with hot reload
npm run build    # Build for production
npm run preview  # Preview production build
```

### Code Style
- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting (recommended)

### Key Components

**Demo.tsx**
- Manages WebSocket connections (live stream + navigation)
- Handles scene selection and object search
- Renders video feed with HUD overlay
- Displays navigation logs and status

**Live Stream Logic**
```typescript
// Persistent connection for always-on video
useEffect(() => {
  const ws = new WebSocket(`${WS_URL}/ws/live_stream`);
  ws.onmessage = (event) => {
    // Update frame when not navigating
    if (!isNavigating) {
      setCurrentFrame(data.data);
    }
  };
}, [serverConnected]);
```

## 🌐 Deployment

### Production Build
```bash
npm run build
```

Build output will be in the `dist/` directory.

### Deploy Options
- **Vercel**: Connect your GitHub repo for automatic deployments
- **Netlify**: Drag and drop the `dist/` folder
- **GitHub Pages**: Use `gh-pages` branch
- **Custom Server**: Serve the `dist/` directory with any static file server

### Backend Deployment
The backend must be deployed separately and accessible to the frontend. Update the `WS_URL` in `Demo.tsx` to point to your production backend.

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
VITE_BACKEND_URL=http://localhost:8000
```

Update `Demo.tsx`:
```typescript
const WS_URL = import.meta.env.VITE_BACKEND_URL || "ws://localhost:8000";
```

## 📊 Performance

- **Frame Rate**: 30 FPS live stream
- **Latency**: <50ms WebSocket round trip (local)
- **Build Size**: ~500KB gzipped
- **Load Time**: <2s on decent connection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **AI2-THOR**: Allen Institute for AI - Photorealistic 3D environments
- **ProcTHOR-RL**: Allenai team for the pretrained navigation model
- **shadcn/ui**: For beautiful UI components
- **Vercel**: For hosting the demo

## 📧 Contact

**Abdelaziz** - [@azizlg](https://github.com/azizlg)

**Project Links**:
- Frontend: [navi-quest-bot](https://github.com/azizlg/navi-quest-bot)
- Backend: [SnapNav](https://github.com/azizlg/SnapNav)

---

Made with ❤️ using React, TypeScript, and AI2-THOR
