import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ExternalLink,
  Navigation,
  Eye,
  CheckCircle2,
  ArrowRight,
  RotateCcw,
  Apple,
  Tv,
  Armchair,
  Coffee,
  AlertCircle,
  Wifi,
  WifiOff
} from "lucide-react";
import { Button } from "@/components/ui/button";

const objects = [
  { id: "Apple", label: "Apple", icon: Apple },
  { id: "Mug", label: "Mug", icon: Coffee },
  { id: "Chair", label: "Chair", icon: Armchair },
  { id: "Television", label: "Television", icon: Tv },
];

// Available AI2-THOR scenes
const scenes = Array.from({ length: 30 }, (_, i) => ({
  id: `FloorPlan${i + 1}`,
  label: `FloorPlan ${i + 1}`,
}));

// Backend WebSocket URL - change this to your backend server URL
const WS_URL = "ws://localhost:8000";

interface NavigationStep {
  action: string;
  status: "pending" | "active" | "complete" | "success";
  confidence?: number;
}

const Demo = () => {
  const [selectedObject, setSelectedObject] = useState<string | null>(null);
  const [selectedScene, setSelectedScene] = useState<string>("FloorPlan1");
  const [isNavigating, setIsNavigating] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [navigationSteps, setNavigationSteps] = useState<NavigationStep[]>([]);
  const [currentAction, setCurrentAction] = useState<string>("");
  const [stepCount, setStepCount] = useState(0);
  const [success, setSuccess] = useState<boolean | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [serverConnected, setServerConnected] = useState(false);
  const [commandInput, setCommandInput] = useState("");
  const [commandError, setCommandError] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const liveWsRef = useRef<WebSocket | null>(null);

  // Check server connection on mount
  useEffect(() => {
    fetch("http://localhost:8000/")
      .then(() => setServerConnected(true))
      .catch(() => setServerConnected(false));
  }, []);

  // Connect to live stream for persistent real-time view
  useEffect(() => {
    if (!serverConnected) return;

    const ws = new WebSocket(`${WS_URL}/ws/live_stream`);
    liveWsRef.current = ws;

    ws.onopen = () => {
      console.log("Live stream connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "frame" && !isNavigating) {
        // Only update from live stream when not navigating
        setCurrentFrame(`data:image/jpeg;base64,${data.data}`);
      }
    };

    ws.onerror = (error) => {
      console.error("Live stream error:", error);
    };

    ws.onclose = () => {
      console.log("Live stream closed");
    };

    return () => {
      if (liveWsRef.current) {
        liveWsRef.current.close();
      }
    };
  }, [serverConnected, isNavigating]);

  const handleCommandSubmit = async () => {
    if (!commandInput.trim()) {
      setCommandError("Please enter a command");
      return;
    }

    setCommandError("");

    // Parse the command to extract target object
    try {
      const response = await fetch(`http://localhost:8000/parse_command/${encodeURIComponent(commandInput)}`);
      const data = await response.json();

      if (data.success) {
        startDemo(data.target);
      } else {
        setCommandError(data.error || "Could not understand command. Try: 'find the apple'");
      }
    } catch (error) {
      setCommandError("Failed to connect to backend server");
    }
  };

  // Auto-load scene in Unity window when scene selection changes
  useEffect(() => {
    if (serverConnected && selectedScene) {
      // Send scene change to live stream
      if (liveWsRef.current && liveWsRef.current.readyState === WebSocket.OPEN) {
        liveWsRef.current.send(JSON.stringify({
          action: "change_scene",
          scene: selectedScene
        }));
      } else {
        // Fallback to HTTP endpoint
        fetch(`http://localhost:8000/preview_scene/${selectedScene}`)
          .then(res => res.json())
          .then(data => {
            if (data.success) {
              console.log(`Unity window loaded: ${selectedScene}`);
            }
          })
          .catch(err => console.error("Scene preview error:", err));
      }
    }
  }, [selectedScene, serverConnected]);


  const startDemo = (objectId: string) => {
    setSelectedObject(objectId);
    setCommandInput("");
    setIsNavigating(true);
    setSuccess(null);
    setErrorMessage("");
    setNavigationSteps([]);
    setCurrentAction("");
    setStepCount(0);

    // Connect to WebSocket with scene parameter
    const ws = new WebSocket(`${WS_URL}/ws/navigate/${objectId}/${selectedScene}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received:", data);

      switch (data.type) {
        case "status":
          setNavigationSteps(prev => [...prev, {
            action: data.message,
            status: "active"
          }]);
          break;

        case "frame":
          setCurrentFrame(`data:image/jpeg;base64,${data.data}`);
          setStepCount(data.step || 0);
          break;

        case "action":
          setCurrentAction(data.action);
          setStepCount(data.step);
          setNavigationSteps(prev => {
            const newSteps = [...prev];
            // Mark previous as complete
            if (newSteps.length > 0) {
              newSteps[newSteps.length - 1].status = "complete";
            }
            // Add new action
            return [...newSteps, {
              action: `${data.action} (confidence: ${(data.confidence * 100).toFixed(0)}%)`,
              status: "active",
              confidence: data.confidence
            }];
          });
          break;

        case "success":
          setSuccess(true);
          setIsNavigating(false);
          setNavigationSteps(prev => {
            const newSteps = [...prev];
            if (newSteps.length > 0) {
              newSteps[newSteps.length - 1].status = "success";
            }
            return [...newSteps, {
              action: data.message,
              status: "success"
            }];
          });
          break;

        case "failure":
          setSuccess(false);
          setIsNavigating(false);
          setErrorMessage(data.message);
          break;

        case "error":
          setErrorMessage(data.message);
          setIsNavigating(false);
          break;
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setErrorMessage("Connection error. Make sure backend server is running.");
      setIsNavigating(false);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setIsNavigating(false);
    };
  };

  const resetDemo = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setSelectedObject(null);
    setIsNavigating(false);
    setCurrentFrame(null);
    setNavigationSteps([]);
    setCurrentAction("");
    setStepCount(0);
    setSuccess(null);
    setErrorMessage("");
  };

  return (
    <section id="demo" className="py-24 relative">
      <div className="container px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            See It <span className="text-gradient">In Action</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Real-time AI2-THOR object navigation powered by ProcTHOR-RL
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            The Unity scene window will appear separately when you start navigation
          </p>

          {/* Server status indicator */}
          <div className="flex items-center justify-center gap-2 mt-4">
            {serverConnected ? (
              <>
                <Wifi className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-500">Backend server online</span>
              </>
            ) : (
              <>
                <WifiOff className="w-4 h-4 text-destructive" />
                <span className="text-sm text-destructive">
                  Backend server offline - Start server to enable live demo
                </span>
              </>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="max-w-4xl mx-auto"
        >
          {/* Interactive Demo */}
          <div className="rounded-2xl bg-card border border-border overflow-hidden">
            {/* Terminal Header */}
            <div className="px-4 py-3 bg-secondary/50 border-b border-border flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-destructive/70" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
                <div className="w-3 h-3 rounded-full bg-green-500/70" />
              </div>
              <span className="text-sm font-mono text-muted-foreground ml-2">
                snapnav-live-demo
              </span>
            </div>

            <div className="p-6 md:p-8">
              <AnimatePresence mode="wait">
                {!selectedObject ? (
                  <motion.div
                    key="selection"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <p className="text-muted-foreground mb-4 font-mono text-sm">
                      <span className="text-primary">$</span> Enter a command to navigate:
                    </p>

                    {/* Scene Selector */}
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-muted-foreground mb-2">
                        Select Scene:
                      </label>
                      <select
                        value={selectedScene}
                        onChange={(e) => setSelectedScene(e.target.value)}
                        disabled={!serverConnected}
                        className="w-full px-4 py-3 rounded-lg bg-background border border-border text-foreground focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {scenes.map((scene) => (
                          <option key={scene.id} value={scene.id}>
                            {scene.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Text Input for Custom Commands */}
                    <div className="mb-6">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={commandInput}
                          onChange={(e) => setCommandInput(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && serverConnected && handleCommandSubmit()}
                          placeholder='Try: "find the apple" or "go to the mug"'
                          disabled={!serverConnected}
                          className="flex-1 px-4 py-3 rounded-lg bg-background border border-border text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
                        />
                        <Button
                          onClick={handleCommandSubmit}
                          disabled={!serverConnected || !commandInput.trim()}
                          className="px-6"
                        >
                          Execute
                          <ArrowRight className="w-4 h-4 ml-2" />
                        </Button>
                      </div>
                      {commandError && (
                        <p className="text-sm text-destructive mt-2">{commandError}</p>
                      )}
                    </div>

                    {/* Quick Select Buttons */}
                    <div className="space-y-2">
                      <p className="text-xs text-muted-foreground">Quick select:</p>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                        {objects.map((obj) => (
                          <motion.button
                            key={obj.id}
                            onClick={() => serverConnected && startDemo(obj.id)}
                            disabled={!serverConnected}
                            whileHover={serverConnected ? { scale: 1.02 } : {}}
                            whileTap={serverConnected ? { scale: 0.98 } : {}}
                            className={`p-3 rounded-lg bg-secondary/50 border border-border transition-all duration-300 group ${serverConnected
                              ? "hover:border-primary/50 hover:bg-secondary cursor-pointer"
                              : "opacity-50 cursor-not-allowed"
                              }`}
                          >
                            <obj.icon className={`w-6 h-6 mx-auto mb-1 transition-colors ${serverConnected
                              ? "text-muted-foreground group-hover:text-primary"
                              : "text-muted-foreground/50"
                              }`} />
                            <span className="text-xs font-medium text-foreground">
                              {obj.label}
                            </span>
                          </motion.button>
                        ))}
                      </div>
                    </div>

                    {!serverConnected && (
                      <div className="mt-6 p-4 rounded-xl bg-destructive/10 border border-destructive/30 flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0" />
                        <div className="text-sm">
                          <p className="font-semibold text-destructive mb-1">Backend server not running</p>
                          <p className="text-muted-foreground">
                            Start the Python backend server to enable the live demo:
                          </p>
                          <code className="block mt-2 p-2 rounded bg-background/50 text-xs">
                            python backend_server.py
                          </code>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key="navigation"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    {/* Command Display */}
                    <div className="mb-6 p-4 rounded-xl bg-background/50 border border-border">
                      <p className="font-mono text-sm">
                        <span className="text-primary">$</span>{" "}
                        <span className="text-muted-foreground">python find_object.py --scene</span>{" "}
                        <span className="text-foreground">{selectedScene}</span>{" "}
                        <span className="text-muted-foreground">--target</span>{" "}
                        <span className="text-foreground">"{selectedObject.toLowerCase()}"</span>
                      </p>
                      {stepCount > 0 && (
                        <p className="font-mono text-xs text-muted-foreground mt-2">
                          Scene: {selectedScene} | Step: {stepCount} | Action: {currentAction}
                        </p>
                      )}
                    </div>

                    {/* Navigation Visualization - Immersive Mode */}
                    <div className="flex flex-col gap-6">
                      {/* Live Agent View Container */}
                      <div className="relative rounded-xl bg-black border border-border overflow-hidden shadow-2xl ring-1 ring-white/10 group">
                        {/* Video Feed */}
                        <div className="aspect-video relative bg-zinc-900">
                          {currentFrame ? (
                            <img
                              src={currentFrame}
                              alt="Agent view"
                              className="w-full h-full object-contain"
                            />
                          ) : (
                            <div className="absolute inset-0 flex items-center justify-center">
                              <div className="flex flex-col items-center gap-4">
                                <motion.div
                                  animate={{ rotate: 360 }}
                                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                >
                                  <Navigation className="w-12 h-12 text-primary/50" />
                                </motion.div>
                                <p className="text-sm text-muted-foreground animate-pulse">Waiting for video stream...</p>
                              </div>
                            </div>
                          )}

                          {/* HUD Overlay - Top Info Bar */}
                          <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/80 to-transparent pointer-events-none flex justify-between items-start">
                            <div className="flex gap-4">
                              <div className="bg-black/50 backdrop-blur border border-white/10 rounded-lg px-3 py-1.5 flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${serverConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                                <span className="text-xs font-mono text-white/90">LIVE FEED</span>
                              </div>
                              <div className="bg-black/50 backdrop-blur border border-white/10 rounded-lg px-3 py-1.5 hidden md:flex items-center gap-2">
                                <Eye className="w-3 h-3 text-blue-400" />
                                <span className="text-xs font-mono text-white/90">TARGET: {selectedObject?.toUpperCase()}</span>
                              </div>
                            </div>
                            <div className="font-mono text-xs text-white/50 bg-black/30 px-2 py-1 rounded">
                              FPS: 30 • HD
                            </div>
                          </div>

                          {/* HUD Overlay - Crosshair */}
                          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
                            <div className="w-8 h-8 border border-white/50 rounded-full flex items-center justify-center">
                              <div className="w-1 h-1 bg-white rounded-full" />
                            </div>
                          </div>

                          {/* HUD Overlay - Bottom Status Area */}
                          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 to-transparent pointer-events-none">
                            <div className="flex justify-between items-end">
                              <div className="space-y-1">
                                <div className="text-xs font-mono text-white/60">CURRENT ACTION</div>
                                <div className="text-lg font-bold text-white tracking-wide uppercase flex items-center gap-2">
                                  {currentAction || "IDLE"}
                                  {currentAction && (
                                    <span className="text-xs font-normal text-primary bg-primary/20 px-1.5 py-0.5 rounded">EXEC</span>
                                  )}
                                </div>
                              </div>

                              <div className="text-right space-y-1">
                                <div className="text-xs font-mono text-white/60">STEPS TAKEN</div>
                                <div className="text-xl font-bold text-white font-mono">{stepCount}</div>
                              </div>
                            </div>
                          </div>

                          {/* Success overlay */}
                          <AnimatePresence>
                            {success && (
                              <motion.div
                                initial={{ opacity: 0, scale: 0.9 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-10"
                              >
                                <div className="text-center p-8 rounded-2xl bg-black/80 border border-green-500/30 shadow-2xl shadow-green-900/20">
                                  <motion.div
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: "spring", damping: 12 }}
                                  >
                                    <div className="w-20 h-20 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-4 ring-1 ring-green-500/50">
                                      <CheckCircle2 className="w-10 h-10 text-green-500" />
                                    </div>
                                  </motion.div>
                                  <h3 className="text-2xl font-bold text-white mb-1">Target Acquired</h3>
                                  <p className="text-green-400 font-mono text-sm">{selectedObject} found in {stepCount} steps</p>
                                  <div className="mt-6 flex justify-center gap-3">
                                    <Button onClick={resetDemo} className="bg-white text-black hover:bg-white/90">
                                      Scan New Target
                                    </Button>
                                  </div>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>

                          {/* Error message */}
                          {errorMessage && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10">
                              <div className="bg-destructive/10 border border-destructive/50 p-6 rounded-xl max-w-md text-center">
                                <AlertCircle className="w-10 h-10 text-destructive mx-auto mb-3" />
                                <h3 className="text-lg font-bold text-white mb-1">Connection Lost</h3>
                                <p className="text-white/70 text-sm mb-4">{errorMessage}</p>
                                <Button onClick={resetDemo} variant="outline" className="border-white/20 text-white hover:bg-white/10 hover:text-white">
                                  Reset System
                                </Button>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Steps Log - Compact View */}
                      <div className="bg-card/50 border border-border rounded-xl p-4 max-h-48 overflow-y-auto font-mono text-sm custom-scrollbar">
                        <div className="flex items-center gap-2 mb-3 text-xs text-muted-foreground uppercase tracking-wider font-semibold sticky top-0 bg-card/95 backdrop-blur py-1 z-10">
                          <div className="w-2 h-2 rounded-full bg-primary" />
                          System Log
                        </div>
                        <div className="space-y-1.5">
                          {navigationSteps.slice().reverse().map((step, index) => (
                            <motion.div
                              key={`${step.action}-${index}`}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              className={`flex items-start gap-2 text-xs py-1 border-b border-border/40 last:border-0 ${step.status === 'success' ? 'text-green-400' : 'text-foreground/80'
                                }`}
                            >
                              <span className="opacity-50">[{new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}]</span>
                              <span>{step.action}</span>
                            </motion.div>
                          ))}
                          {navigationSteps.length === 0 && (
                            <div className="text-muted-foreground/40 italic text-xs py-2">System initialized. Waiting for commands...</div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Quick start */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-12 p-6 rounded-2xl bg-card/50 border border-border"
          >
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="text-gradient">Quick Start</span>
            </h3>
            <div className="space-y-3 font-mono text-sm">
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <span className="text-muted-foreground"># Clone the repository</span>
                <br />
                <span className="text-foreground">git clone https://github.com/azizlg/SnapNav.git</span>
              </div>
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <span className="text-muted-foreground"># Install dependencies</span>
                <br />
                <span className="text-foreground">pip install -r requirements.txt</span>
              </div>
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <span className="text-muted-foreground"># Run the backend server</span>
                <br />
                <span className="text-foreground">python backend_server.py</span>
              </div>
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <span className="text-muted-foreground"># Or run agent directly</span>
                <br />
                <span className="text-foreground">python find_object.py</span>
              </div>
            </div>

            <Button
              variant="outline"
              className="mt-6 border-border bg-secondary/50 hover:bg-secondary"
              asChild
            >
              <a href="https://github.com/azizlg/SnapNav" target="_blank" rel="noopener noreferrer">
                View Full Documentation
                <ExternalLink className="w-4 h-4 ml-2" />
              </a>
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Demo;
