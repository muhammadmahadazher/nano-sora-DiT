import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors
    c_input = '#34d399' # Green
    c_embed = '#60a5fa' # Blue
    c_block = '#a78bfa' # Purple
    c_cond = '#fbbf24'  # Amber
    c_output = '#f87171' # Red
    c_text = '#1e293b'

    def draw_box(x, y, w, h, label, color, alpha=0.8):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1", 
                                      linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, color='white', weight='bold', 
                ha='center', va='center', fontsize=10)

    # 1. Input Section
    draw_box(5, 75, 15, 10, "Input Image / Noise\n(B, C, H, W)", c_input)
    
    # 2. Patchify & Embed
    draw_box(5, 55, 15, 8, "Patchify (Rearrange)\n(B, N, P*P*C)", c_embed)
    draw_box(5, 40, 15, 8, "Linear Projection\n+ Positional Embed", c_embed)

    # 3. Conditioning Section
    draw_box(30, 85, 15, 8, "Timestep (t)\nSinusoidal MLP", c_cond)
    draw_box(50, 85, 15, 8, "Label (y)\nEmbedding", c_cond)
    draw_box(40, 72, 15, 6, "Joint Conditioning (c)\n(t_emb + y_emb)", c_cond)

    # 4. DiT Block (Simplified)
    # Background for the block loop
    loop_rect = patches.Rectangle((28, 15), 44, 52, linewidth=2, edgecolor=c_block, facecolor=c_block, alpha=0.1, linestyle='--')
    ax.add_patch(loop_rect)
    ax.text(50, 64, "DiT Block (x N Layers)", color=c_block, weight='bold', ha='center')

    # Internal components
    draw_box(32, 50, 12, 6, "adaLN-Zero (1)", c_block)
    draw_box(32, 40, 12, 6, "Multi-Head\nSelf-Attention", c_block)
    draw_box(52, 50, 12, 6, "adaLN-Zero (2)", c_block)
    draw_box(52, 40, 12, 6, "Feed-Forward\n(MLP)", c_block)
    
    # 5. Output Section
    draw_box(80, 55, 15, 8, "Final Layer\n(adaLN + Linear)", c_output)
    draw_box(80, 40, 15, 8, "Unpatchify\n(Rearrange)", c_output)
    draw_box(80, 20, 15, 10, "Predicted Velocity (v)\n(B, C, H, W)", c_output)

    # Arrows - Flow Matching Data Path
    ax.annotate('', xy=(12.5, 65), xytext=(12.5, 75), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    ax.annotate('', xy=(12.5, 50), xytext=(12.5, 55), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    ax.annotate('', xy=(28, 45), xytext=(20, 45), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    
    # Arrow - Conditioning Path
    ax.annotate('', xy=(40, 78), xytext=(37.5, 85), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    ax.annotate('', xy=(40, 78), xytext=(57.5, 85), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    
    # Inside DiT Block arrows
    ax.annotate('', xy=(38, 46), xytext=(38, 50), arrowprops=dict(arrowstyle='->', lw=1, color=c_text))
    ax.annotate('', xy=(48, 43), xytext=(44, 43), arrowprops=dict(arrowstyle='->', lw=1, color=c_text))
    ax.annotate('', xy=(58, 46), xytext=(58, 50), arrowprops=dict(arrowstyle='<-', lw=1, color=c_text))
    
    # Conditioning to blocks
    ax.annotate('', xy=(38, 56), xytext=(40, 72), arrowprops=dict(arrowstyle='->', lw=1, color=c_cond, ls=':'))
    ax.annotate('', xy=(58, 56), xytext=(55, 72), arrowprops=dict(arrowstyle='->', lw=1, color=c_cond, ls=':'))

    # Final arrows
    ax.annotate('', xy=(80, 59), xytext=(72, 45), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    ax.annotate('', xy=(87.5, 48), xytext=(87.5, 55), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))
    ax.annotate('', xy=(87.5, 30), xytext=(87.5, 40), arrowprops=dict(arrowstyle='->', lw=2, color=c_text))

    # Flow Matching / ODE Logic
    draw_box(40, 5, 20, 8, "Euler Solver\nx_t+dt = x_t + v * dt", '#64748b')
    ax.annotate('ODE Integration', xy=(50, 15), xytext=(80, 25), 
                arrowprops=dict(arrowstyle='<-', lw=1, color='#64748b', connectionstyle="arc3,rad=.2"))

    plt.title("Nano-Sora Architecture: DiT + Flow Matching (Rectified Flow)", fontsize=18, weight='bold', pad=20, color='#0f172a')
    plt.text(50, -5, "Visualized for: RTX 4060 | Precision: FP16 | Model: facebook/DiT-XL-2-256", 
             ha='center', fontsize=12, color='#64748b', style='italic')

    plt.tight_layout()
    plt.savefig('nanosora_architecture.png', bbox_inches='tight', transparent=False, facecolor='white')
    print("Architecture diagram saved as nanosora_architecture.png")

if __name__ == "__main__":
    create_diagram()