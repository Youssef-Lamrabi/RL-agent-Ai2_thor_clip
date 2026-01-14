import { Github, ExternalLink } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 border-t border-border">
      <div className="container px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold">
              <span className="text-foreground">Snap</span>
              <span className="text-gradient">Nav</span>
            </span>
            <span className="text-muted-foreground text-sm">
              Object Navigation Agent
            </span>
          </div>

          <div className="flex items-center gap-6">
            <a
              href="https://github.com/azizlg/SnapNav"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2"
            >
              <Github className="w-5 h-5" />
              GitHub
            </a>
            <a
              href="https://ai2thor.allenai.org/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors flex items-center gap-2"
            >
              AI2-THOR
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
          <p>Built with PyTorch, OpenAI CLIP & AI2-THOR</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
