import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_modern_diagram():
    # Setup Figure (Ultra-Dark Mode)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')

    # Color Palette
    c_accent = '#00F2FF'  # Cyan Glow
    c_purple = '#7000FF'
    c_pink = '#FF007A'
    c_text = '#FFFFFF'
    c_sub = '#888888'

    def draw_node(x, y, w, h, title, sub, color):
        # Glow layers
        for i in range(1, 8):
            glow = patches.Rectangle((x-i*0.4, y-i*0.4), w+i*0.8, h+i*0.8, color=color, alpha=0.02, zorder=1)
            ax.add_patch(glow)
        
        # Main Node Slab
        node = patches.Rectangle((x, y), w, h, facecolor='#111111', edgecolor=color, linewidth=2, zorder=3)
        ax.add_patch(node)
        
        # Top line accent
        ax.add_patch(patches.Rectangle((x, y+h-1), w, 1, facecolor=color, alpha=0.9, zorder=4))

        # Text labels (Single line strings)
        ax.text(x+w/2, y+h/2+1.5, title, ha='center', va='center', color=c_text, fontsize=11, fontweight='bold', zorder=5)
        ax.text(x+w/2, y+h/2-2, sub, ha='center', va='center', color=c_sub, fontsize=8, zorder=5)

    # 1. INPUT
    draw_node(5, 18, 15, 12, "LATENT INPUT", "Gaussian Noise xT", c_accent)

    # 2. TOKENIZER
    draw_node(26, 18, 15, 12, "PATCH EMBED", "Space-Time Projection", c_accent)

    # 3. BACKBONE
    # Translucent container for the blocks
    ax.add_patch(patches.Rectangle((46, 14), 28, 20, facecolor='#151515', alpha=0.3, zorder=2))
    ax.text(60, 31, "DIT BACKBONE (L=12)", ha='center', color=c_purple, fontsize=10, fontweight='bold', zorder=5)

    for i in range(3):
        draw_node(48 + i*8.5, 18, 7, 12, f"DiT-{i+1}", "adaLN-0", c_purple)

    # 4. PREDICTOR
    draw_node(80, 18, 15, 12, "OUTPUT HEAD", "Velocity Predictor (v)", c_pink)

    # 5. CONDITIONING
    draw_node(48, 40, 10, 6, "TIMESTEP", "MLP Encoding", '#FFD700')
    draw_node(62, 40, 10, 6, "LABEL", "Embedding Table", '#FFD700')

    # --- CONNECTORS ---
    def draw_path(x1, y1, x2, y2, color):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='simple, head_width=6, head_length=6', color=color, alpha=0.8))

    # Flow
    draw_path(20, 24, 26, 24, c_accent)
    draw_path(41, 24, 46, 24, c_accent)
    draw_path(76, 24, 80, 24, c_pink)
    
    # Condition
    draw_path(53, 40, 53, 33, '#FFD700')
    draw_path(67, 40, 67, 33, '#FFD700')

    # ODE ARC (Simplified plotting)
    arc_x = np.linspace(12, 88, 50)
    arc_y = 6 - 4 * np.sin(np.pi * (arc_x - 12) / 76)
    ax.plot(arc_x, arc_y, color=c_accent, lw=1.5, alpha=0.2, linestyle='dotted')
    ax.text(50, 4, "ITERATIVE FLOW SOLVER (ODE LOOP)", ha='center', color=c_sub, fontsize=9, fontweight='bold')

    # Branding
    plt.text(5, 46, "NANO-SORA", fontsize=36, color='white', fontweight='black')
    plt.text(5, 42, "ARCHITECTURAL BLUEPRINT // DIT + RECTIFIED FLOW", fontsize=11, color=c_accent, fontweight='bold')

    plt.tight_layout()
    plt.savefig('nanosora_blueprint.png', facecolor='#050505', bbox_inches='tight')
    print("Blueprint saved as nanosora_blueprint.png")

if __name__ == "__main__":
    create_modern_diagram()