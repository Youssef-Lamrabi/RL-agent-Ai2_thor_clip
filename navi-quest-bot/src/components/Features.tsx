import { motion } from "framer-motion";
import { MessageSquare, Brain, Target, RefreshCw, Gamepad2, Zap } from "lucide-react";

const features = [
  {
    icon: MessageSquare,
    title: "Natural Language Search",
    description: "Simply tell the agent what to find using everyday language commands.",
  },
  {
    icon: Brain,
    title: "CLIP Visual Encoding",
    description: "Leverages ResNet50-based CLIP model for powerful vision-language understanding.",
  },
  {
    icon: Target,
    title: "16 Object Types",
    description: "Supports Apple, Mug, Chair, Television, and many more common objects.",
  },
  {
    icon: RefreshCw,
    title: "Stuck Detection",
    description: "Intelligent recovery mechanisms when the agent encounters obstacles.",
  },
  {
    icon: Gamepad2,
    title: "Real-time Navigation",
    description: "Interactive visualization of the agent's navigation in 3D environments.",
  },
  {
    icon: Zap,
    title: "Fast Fine-tuning",
    description: "Quickly adapt the model to specific scenes with minimal training.",
  },
];

const Features = () => {
  return (
    <section className="py-24 relative">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/20 to-background" />
      
      <div className="container relative z-10 px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Powerful <span className="text-gradient">Features</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Built with state-of-the-art AI technologies for intelligent object navigation
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="group p-6 rounded-2xl bg-card/50 border border-border backdrop-blur-sm hover:border-primary/50 transition-all duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-primary flex items-center justify-center mb-4 group-hover:glow-primary transition-all duration-300">
                <feature.icon className="w-6 h-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-foreground">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
