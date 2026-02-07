import streamlit as st
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from PIL import Image
import io

# ==========================================
# 1. APPLICATION CONFIGURATION & STYLING
# ==========================================

# Set page configuration for a cinematic wide layout
st.set_page_config(
    page_title="Nano-Sora: Premium DiT Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS
# This injects a premium dark theme with translucent glass cards, neon accents, and smooth transitions.
glass_css = """
<style>
    /* -------------------------------------------------------------------------- */
    /*                                GLOBAL THEME                                */
    /* -------------------------------------------------------------------------- */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main background: Deep Cosmic Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #000000 90%);
        color: #e2e8f0;
    }

    /* -------------------------------------------------------------------------- */
    /*                                SIDEBAR STYLING                             */
    /* -------------------------------------------------------------------------- */
    
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Input Labels */
    .stSidebar [data-testid="stMarkdownContainer"] p {
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* -------------------------------------------------------------------------- */
    /*                                GLASS CARDS                                 */
    /* -------------------------------------------------------------------------- */
    
    /* Targeting vertical blocks to create glass containers */
    div[data-testid="stVerticalBlock"] > div.element-container {
        
    }

    /* Custom class for glass containers we will inject */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }

    /* -------------------------------------------------------------------------- */
    /*                                COMPONENT STYLING                           */
    /* -------------------------------------------------------------------------- */

    /* Premium Gradient Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }

    /* Titles */
    h1 {
        background: linear-gradient(to right, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #cbd5e1;
        font-weight: 400;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.02) !important;
        border-radius: 10px;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #3b82f6;
    }

</style>
"""
st.markdown(glass_css, unsafe_allow_html=True)

# ==========================================
# 2. MODEL PIPELINE (CACHED)
# ==========================================

@st.cache_resource
def get_pipeline():
    """
    Loads the facebook/DiT-XL-2-256 model pipeline.
    Uses cached loading to prevent reloading on every interaction.
    Optimized for FP16 precision on CUDA.
    """
    model_id = "facebook/DiT-XL-2-256"
    
    # Load the Diffusion Transformer Pipeline
    pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    
    # Switch to DPMSolver for faster inference (20 steps vs 50+)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU
    pipe = pipe.to("cuda")
    return pipe

# ==========================================
# 3. MAIN APPLICATION LOGIC
# ==========================================

def main():
    # Header Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("🌌 Nano-Sora: Premium DiT Explorer")
    st.markdown("### Interactive Diffusion Transformer Visualization")
    st.markdown("""
    Experience the state-of-the-art **DiT-XL/2** architecture. This application uses a **Diffusion Transformer** 
    backbone combined with **Flow Matching** principles (approximated here via DPMSolver) to generate 
    high-fidelity images conditioned on ImageNet classes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SIDEBAR: CONTROL PANEL
    # -----------------------------------------------------------------------
    st.sidebar.header("🎨 Generation Studio")
    
    # A curated subset of the 1000 ImageNet classes for easy exploration
    class_map = {
        "Tench (Fish)": 0,
        "Goldfish": 1,
        "Great White Shark": 2,
        "Hammerhead Shark": 4,
        "Ostrich": 9,
        "Golden Retriever": 207,
        "Corgi": 263,
        "Siamese Cat": 284,
        "Tiger": 292,
        "Monarch Butterfly": 323,
        "Airplane": 404,
        "Space Shuttle": 812,
        "Fire Engine": 555,
        "Espresso": 967,
        "Volcano": 980
    }
    
    # Selection Inputs
    selected_name = st.sidebar.selectbox("Subject (Class Condition)", list(class_map.keys()), index=11)
    class_id = class_map[selected_name]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Hyperparameters")
    
    steps = st.sidebar.slider("Inference Steps", 10, 50, 25, help="More steps = higher quality, slower generation.")
    guidance = st.sidebar.slider("Guidance Scale (CFG)", 1.0, 10.0, 4.0, help="Higher values adhere closer to the text/class prompt.")
    
    st.sidebar.markdown("---")
    
    # Advanced Details Expander
    with st.sidebar.expander("📝 Model Details"):
        st.markdown(f"""
        **Model:** facebook/DiT-XL-2-256
        **Params:** 675M
        **Resolution:** 256x256
        **Precision:** FP16
        **Scheduler:** DPM-Solver++
        """)

    # -----------------------------------------------------------------------
    # MAIN WORKSPACE
    # -----------------------------------------------------------------------
    
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 Configuration Scope")
        st.info(f"**Target Class:** {selected_name}")
        st.info(f"**Class ID:** {class_id}")
        st.info(f"**Solver Steps:** {steps}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Generation Trigger
        generate_btn = st.button("✨ Generate Masterpiece")

    # -----------------------------------------------------------------------
    # INFERENCE EXECUTION
    # -----------------------------------------------------------------------
    if generate_btn:
        pipe = get_pipeline()
        
        # UI Feedback
        progress_text = "✨ Denoising latent patches..."
        my_bar = st.progress(0, text=progress_text)
        
        # Execute Generation
        with torch.inference_mode():
            # Note: diffusers pipeline handles the loop, we simulate progress for UX or use callback if needed.
            # For this demo, standard spinner is smoother.
            with st.spinner("🖌️ The DiT is articulating the latent space..."):
                output = pipe(
                    class_labels=[class_id],
                    num_inference_steps=steps,
                    guidance_scale=guidance
                )
                my_bar.progress(100, text="Generation Complete!")
                
            image = output.images[0]
            
            with col2:
                st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption=f"Generated {selected_name} | {steps} Steps")
                
                # Buffer for download
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                # Download Action
                st.download_button(
                    label="📥 Download Artwork",
                    data=byte_im,
                    file_name=f"{selected_name.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.balloons()
    
    elif not generate_btn:
        # Placeholder state
        with col2:
            st.markdown('<div class="glass-card" style="display: flex; justify-content: center; align-items: center; height: 300px; color: #64748b;">', unsafe_allow_html=True)
            st.markdown("*(Select a class and click Generate to visualize)*")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
