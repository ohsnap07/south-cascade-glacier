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

    /* Hide the min/max range numbers below the slider track */
    [data-testid="stSliderTickBar"],
    [data-testid="stTickBar"] {
        display: none !important;
    }

    /* Dark gray sidebar heading ("Climate Parameters") and slider titles */
    [data-testid="stSidebar"] h3,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #333333 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Render Title ---
st.markdown('<div class="retro-title">South Cascade Glacier</div>', unsafe_allow_html=True)

# --- Load Basemap & Rasters ---
@st.cache_data
def load_basemap():
    return Image.open("SouthCascades.png")

@st.cache_data
def load_rasters():
    with rasterio.open("SouthCascadesBEDROCK.tif") as src_bed:
        bedrock = src_bed.read(1).astype(float)
        bounds = src_bed.bounds  # (left, bottom, right, top)
        if src_bed.nodata is not None:
            bedrock[bedrock == src_bed.nodata] = np.nan

    with rasterio.open("thickness.tif") as src_thick:
        thickness = src_thick.read(1).astype(float)
        if src_thick.nodata is not None:
            thickness[thickness == src_thick.nodata] = np.nan

    # Clean out zero or negative noise so only actual ice remains
    thickness[thickness <= 0] = np.nan
    return bedrock, thickness, bounds

def compute_glacier_geometry(bedrock, base_thickness, ela, base_ela=1875.0):
    adj_thickness = base_thickness.copy()

    # --- PLACEHOLDER FOR FUTURE EXTENT / GEOMETRY UPDATES ---
    # Modify adj_thickness here when modeling advance/retreat
    # adj_thickness[adj_thickness <= 0] = np.nan
    # --------------------------------------------------------

    ice_mask = ~np.isnan(adj_thickness) & (adj_thickness > 0)
    surface_dem = np.where(ice_mask, bedrock + adj_thickness, np.nan)
    return adj_thickness, surface_dem, ice_mask

# 1. Load basemap, bedrock DEM, and glacier thickness DEM
basemap_img = load_basemap()
bedrock, base_thickness, bounds = load_rasters()
map_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# 2. Sidebar Controls & ELA Calculation
st.sidebar.markdown("### Climate Parameters")
delta_t = st.sidebar.slider("Temperature Change (°C)", min_value=-2.0, max_value=4.0, value=0.0, step=0.2)
delta_p = st.sidebar.slider("Precipitation Change (%)", min_value=-40, max_value=40, value=0, step=5)

base_ela = 1875.0
lapse_rate = 0.0065
new_ela = base_ela + (delta_t / lapse_rate) - (delta_p * 2.5)

st.sidebar.metric("ELA", f"{new_ela:.0f} m", f"{new_ela - base_ela:+.0f} m")

# 3. Compute glacier geometry & masked surface DEM
adj_thickness, surface_dem, ice_mask = compute_glacier_geometry(
    bedrock, base_thickness, new_ela, base_ela=base_ela
)

# 4. Render Map
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

# Basemap
ax.imshow(basemap_img, extent=map_extent, origin="upper")

# Semi-transparent glacier thickness overlay
if np.any(ice_mask):
    ax.imshow(
        adj_thickness,
        extent=map_extent,
        origin="upper",
        cmap="Blues",
        alpha=0.55,
        vmin=0,
        vmax=np.nanmax(base_thickness),
    )

    # Crisp glacier margin outline
    ax.contour(
        ice_mask.astype(int),
        levels=[0.5],
        colors="#003366",
        linewidths=1.0,
        extent=map_extent,
        origin="upper",
    )

    # ELA contour constrained strictly to the glacier footprint
    if np.nanmin(surface_dem) <= new_ela <= np.nanmax(surface_dem):
        ax.contour(
            surface_dem,
            levels=[new_ela],
            colors="crimson",
            linewidths=2.2,
            extent=map_extent,
            origin="upper",
        )

ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
st.pyplot(fig, use_container_width=True)