import { motion } from "framer-motion";
import { Github, Play, Navigation, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Grid background */}
      <div className="absolute inset-0 grid-pattern opacity-40" />
      
      {/* Gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-primary/20 blur-[120px] animate-pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 rounded-full bg-accent/20 blur-[100px] animate-pulse-glow" style={{ animationDelay: '1.5s' }} />
      
      {/* Floating navigation icon */}
      <motion.div
        className="absolute top-1/3 right-[15%] hidden lg:block"
        animate={{ y: [0, -20, 0], rotate: [0, 5, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      >
        <div className="p-4 rounded-2xl bg-secondary/50 border border-border backdrop-blur-sm glow-primary">
          <Navigation className="w-12 h-12 text-primary" />
        </div>
      </motion.div>

      <div className="container relative z-10 px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center max-w-4xl mx-auto"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border mb-8"
          >
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-muted-foreground">
              Powered by CLIP & AI2-THOR
            </span>
          </motion.div>

          {/* Main heading */}
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
            <span className="text-foreground">Snap</span>
            <span className="text-gradient">Nav</span>
          </h1>

          <p className="text-xl md:text-2xl text-muted-foreground mb-4 font-medium">
            Object Navigation Agent
          </p>

          <p className="text-lg text-muted-foreground/80 max-w-2xl mx-auto mb-10 leading-relaxed">
            A fine-tuned ProcTHOR-RL agent that finds objects in 3D environments 
            using natural language. Just say{" "}
            <span className="font-mono text-primary">"find the apple"</span>{" "}
            and watch it navigate.
          </p>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Button
              size="lg"
              className="bg-gradient-primary text-primary-foreground hover:opacity-90 transition-opacity glow-primary text-base px-8 py-6"
              asChild
            >
              <a href="https://github.com/azizlg/SnapNav" target="_blank" rel="noopener noreferrer">
                <Github className="w-5 h-5 mr-2" />
                View on GitHub
              </a>
            </Button>
            
            <Button
              size="lg"
              variant="outline"
              className="border-border bg-secondary/50 hover:bg-secondary text-foreground text-base px-8 py-6"
              asChild
            >
              <a href="#demo">
                <Play className="w-5 h-5 mr-2" />
                Watch Demo
              </a>
            </Button>
          </motion.div>

          {/* Tech badges */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="mt-16 flex flex-wrap justify-center gap-3"
          >
            {["PyTorch", "OpenAI CLIP", "AI2-THOR", "Python 3.7+"].map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 rounded-lg bg-secondary/30 border border-border text-sm font-mono text-muted-foreground"
              >
                {tech}
              </span>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
