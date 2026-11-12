import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="South Cascade Glacier",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* Import Arimo from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Arimo:wght@400;700&display=swap');

    /* Override Streamlit's primary color variable at the .stApp level */
    .stApp {
        --primary-color: #0054E3 !important;
        background-color: #FFFFFF;
        color: #000000;
    }

    /* Force Arimo everywhere */
    html, body, [class*="css"], .stApp, [data-testid="stSidebar"], 
    h1, h2, h3, p, label, div, span {
        font-family: 'Arimo', Arial, sans-serif !important;
    }

    /* Completely remove top header bar, Deploy button, settings menu, and sidebar arrows */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Classic Windows 2000 / XP light grey sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #808080;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }

    /* Full-bleed wide layout padding */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }

    /* Early-2000s Title Styling in Arimo */
    .retro-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #000000;
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
        padding-bottom: 0.3rem;
    }

    .retro-subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #000000;
        margin-top: 0.3rem;
        margin-bottom: 0.8rem;
    }


    /* Metric label ("ELA") */
    [data-testid="stMetricLabel"] p {
        font-size: 0.95rem !important;
        font-weight: 400 !important;
        color: #000000 !important;
    }

    /* Metric value ("1875 m") - matched to the exact same size as "ELA" */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        line-height: 1.2 !important;
    }

    /* --- CUSTOM SLIDER COLORS --- */
    div[data-baseweb="slider"] div[style*="background"] {
        background-image: none !important;
        background-color: #0054E3 !important;
    }

    div[data-baseweb="slider"] > div > div {
        background-color: #B8B8B8 !important;
    }

    div[data-baseweb="slider"] [role="slider"] {
        background-color: #000000 !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 2px !important;
        box-shadow: 1px 1px 0px #000000 !important;
    }

    [data-testid="stSliderThumbValue"],
    [data-testid="stThumbValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    /* Hide the min/max range numbers below the slider track */
    [data-testid="stSliderTickBar"],
    [data-testid="stTickBar"] {
    display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Render Title ---
st.markdown('<div class="retro-title">South Cascade Glacier</div>', unsafe_allow_html=True)

# --- Sidebar Sliders ---
st.sidebar.markdown("### Climate Parameters")
delta_t = st.sidebar.slider("Temperature Change (°C)", min_value=-2.0, max_value=4.0, value=0.0, step=0.2)
delta_p = st.sidebar.slider("Precipitation Change (%)", min_value=-40, max_value=40, value=0, step=5)

# --- Load Basemap & DEM ---
@st.cache_data
def load_data():
    basemap = Image.open("SouthCascades.png")
    
    with rasterio.open("SouthCascadesBEDROCK.tif") as src:
        dem = src.read(1).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
    return basemap, dem, extent

basemap_img, dem_data, map_extent = load_data()

# --- Glaciology Calculations ---
BASE_ELA = 1875        # Baseline ELA (meters)
LAPSE_RATE = 0.0065    # 6.5°C per 1,000 m
PRECIP_FACTOR = 50     # ~50 m shift per 10% precip change

new_ela = BASE_ELA + (delta_t / LAPSE_RATE) - ((delta_p / 100.0) * PRECIP_FACTOR)
st.sidebar.metric("ELA", f"{int(new_ela)} m")

# Get the exact native pixel dimensions of your basemap PNG
img_w, img_h = basemap_img.size
native_dpi = 150

# Set figsize so (figsize * native_dpi) equals your exact image pixel dimensions
fig, ax = plt.subplots(figsize=(img_w / native_dpi, img_h / native_dpi), dpi=native_dpi)
fig.patch.set_facecolor("#FFFFFF")

ax.imshow(basemap_img, extent=map_extent)

ax.contour(
    dem_data, 
    levels=[new_ela], 
    colors=["crimson"], 
    linewidths=2.5, 
    linestyles="solid", 
    extent=map_extent, 
    origin="upper"
)

ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Render at exact 1:1 pixel resolution
st.pyplot(fig, width="content")