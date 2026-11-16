import io
import os
import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
import json

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

    /* Completely remove top header bar, Deploy button, settings menu, sidebar arrows, and image fullscreen button */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stFullScreenFrame"] button,
    button[title="View fullscreen"] {
        display: none !important;
    }

    /* Classic Windows 2000 / XP white sidebar with dark gray border */
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

    /* Boxy retro slider handle with permanent drop shadow and no focus glow */
    div[data-baseweb="slider"] [role="slider"],
    div[data-baseweb="slider"] [role="slider"]:hover,
    div[data-baseweb="slider"] [role="slider"]:focus,
    div[data-baseweb="slider"] [role="slider"]:active,
    div[data-baseweb="slider"] [role="slider"]:focus-visible {
        background-color: #000000 !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 2px !important;
        box-shadow: 2px 2px 0px #808080 !important;
        outline: none !important;
    }

    /* Strip focus outlines on wrapper divs WITHOUT wiping out the thumb's own shadow */
    div[data-baseweb="slider"] div:not([role="slider"]):focus,
    div[data-baseweb="slider"] div:not([role="slider"]):focus-visible {
        outline: none !important;
        box-shadow: none !important;
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

    /* Hide the delta indicator (+0 m / arrow) under the ELA metric */
    [data-testid="stMetricDelta"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    import json
    from scipy.ndimage import label

    adj_thickness = base_thickness.copy()

    RHO_ICE = 917.0        # Bulk glacier ice density (kg / m^3)
    G = 9.81               # Gravitational acceleration (m / s^2)
    ELEV_SPAN = 440.0      # Approx vertical relief of South Cascade Glacier (~1640 m to 2080 m)

    initial_mask = ~np.isnan(base_thickness) & (base_thickness > 0)

    if np.any(initial_mask):
        nrows, ncols = base_thickness.shape
        xmin, xmax, ymin, ymax = map_extent
        dx = (xmax - xmin) / float(ncols)
        dy = (ymax - ymin) / float(nrows)

        # Build UTM coordinate grids (EPSG:32610) for every pixel
        x_utm = xmin + (np.arange(ncols) + 0.5) * dx
        y_utm = ymax - (np.arange(nrows) + 0.5) * dy
        X_grid, Y_grid = np.meshgrid(x_utm, y_utm)

        # 1. Load manual 2D centerline from QGIS GeoJSON export (located alongside app.py)
        with open("centerline.geojson") as f:
            gj = json.load(f)
        coords = gj["features"][0]["geometry"]["coordinates"]
        if isinstance(coords[0][0], list):
            coords = coords[0]
        pts = np.array([[c[0], c[1]] for c in coords], dtype=float)

        seg_vecs = pts[1:] - pts[:-1]
        seg_lens = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])
        cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])

        # Automatically detect L_0 where the drawn centerline crosses the modern terminus edge
        pt_r = np.clip(((ymax - pts[:, 1]) / dy).astype(int), 0, nrows - 1)
        pt_c = np.clip(((pts[:, 0] - xmin) / dx).astype(int), 0, ncols - 1)
        on_ice = initial_mask[pt_r, pt_c]
        last_ice_idx = int(np.where(on_ice)[0][-1]) if np.any(on_ice) else len(pts) - 1
        L_0 = float(cum_s[last_ice_idx])

        # 2. Project every raster pixel onto the centerline -> along-flow `s` & cross-valley `d_perp`
        best_dist2 = np.full(base_thickness.shape, np.inf)
        s_grid = np.zeros(base_thickness.shape)
        for i in range(len(seg_lens)):
            vx, vy = seg_vecs[i]
            L2 = max(seg_lens[i] ** 2, 1e-6)
            t = ((X_grid - pts[i, 0]) * vx + (Y_grid - pts[i, 1]) * vy) / L2
            t_clamped = np.clip(t, 0.0, 1.0)
            proj_x = pts[i, 0] + t_clamped * vx
            proj_y = pts[i, 1] + t_clamped * vy
            d2 = (X_grid - proj_x) ** 2 + (Y_grid - proj_y) ** 2
            closer = d2 < best_dist2
            best_dist2[closer] = d2[closer]
            s_grid[closer] = cum_s[i] + t_clamped[closer] * seg_lens[i]

        d_perp = np.sqrt(best_dist2)

        # 3. Compute slope-adjusted effective basal shear stress: tau_eff = rho * g * h * sin(alpha)
        base_surface = bedrock + np.nan_to_num(base_thickness, nan=0.0)
        dz_dy, dz_dx = np.gradient(base_surface, dy, dx)
        sin_alpha = np.hypot(dz_dx, dz_dy) / np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
        tau_eff = np.clip(RHO_ICE * G * np.nan_to_num(base_thickness, nan=0.0) * sin_alpha, 4.0e4, 1.6e5)
        tau_mean = float(np.nanmean(tau_eff[initial_mask]))

        # 4. Solve for L_new directly where parabolic centerline height h(s) hits 0
        coeff_2d = (2.0 * tau_eff) / (RHO_ICE * G)
        coeff_center = (2.0 * tau_mean) / (RHO_ICE * G)
        H_0 = np.sqrt(coeff_center * L_0)

        # Relief-scaled thickness coupling to ELA shift
        delta_ela = ela - base_ela
        relief_factor = np.maximum(1.0 - 2.2 * (delta_ela / ELEV_SPAN), 0.0)
        H_new = H_0 * np.sqrt(relief_factor)

        # Exact length where h(s) = sqrt(coeff_center * (L_new - s)) reaches 0, capped at centerline end
        L_new = (H_new ** 2) / max(coeff_center, 1e-6)
        L_new = float(np.clip(L_new, 0.0, cum_s[-1]))

        h_0 = np.sqrt(np.maximum(coeff_2d * (L_0 - s_grid), 0.0))
        h_new = np.sqrt(np.maximum(coeff_2d * (L_new - s_grid), 0.0))
        delta_h = h_new - h_0

        # 5. Apply perturbation inside initial_mask and grow tapered advancing tongue along centerline
        max_half_width = float(np.nanpercentile(d_perp[initial_mask], 95))

        # Taper valley half-width parabolically down to zero at the new terminus (s = L_new)
        taper_factor = np.sqrt(np.clip((L_new - s_grid) / max(L_new - L_0 * 0.85, 1.0), 0.0, 1.0))
        local_half_width = max_half_width * taper_factor
        in_valley_corridor = (s_grid >= 0.0) & (s_grid < L_new) & (d_perp <= local_half_width)

        # Sample centerline bed elevation along s and apply parabolic cross-valley dome
        center_r = np.clip(((ymax - pts[:, 1]) / dy).astype(int), 0, nrows - 1)
        center_c = np.clip(((pts[:, 0] - xmin) / dx).astype(int), 0, ncols - 1)
        bed_along_s = np.interp(s_grid, cum_s, bedrock[center_r, center_c])

        cross_dome = np.clip(1.0 - (d_perp / np.maximum(local_half_width, 1.0)) ** 2, 0.0, 1.0)
        h_parabolic_new = np.sqrt(np.maximum(coeff_center * (L_new - s_grid), 0.0)) * np.sqrt(cross_dome)
        advance_surf = bed_along_s + h_parabolic_new
        advance_thick = advance_surf - bedrock

        adj_thickness = np.where(
            initial_mask,
            base_thickness + delta_h,
            np.where(in_valley_corridor & (s_grid >= L_0 * 0.85), advance_thick, np.nan),
        )
        adj_thickness[(adj_thickness <= 0) | (s_grid >= L_new)] = np.nan

        # Keep only the single contiguous glacier connected to the main body
        valid = ~np.isnan(adj_thickness) & (adj_thickness > 0)
        labeled, num_features = label(valid)
        if num_features > 1:
            main_labels = np.unique(labeled[initial_mask & valid])
            adj_thickness[~np.isin(labeled, main_labels)] = np.nan

    ice_mask = ~np.isnan(adj_thickness) & (adj_thickness > 0)
    surface_dem = np.where(ice_mask, bedrock + adj_thickness, np.nan)
    return adj_thickness, surface_dem, ice_mask

# 1. Load basemap, bedrock DEM, and glacier thickness DEM
basemap_img = load_basemap()
bedrock, base_thickness, bounds = load_rasters()
map_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# 2. Sidebar Controls & ELA Calculation
st.sidebar.markdown("### South Cascade Glacier")
temp = st.sidebar.slider("Average Annual Air Temperature at ELA (°C)", min_value=0.0, max_value=4.0, value=1.6, step=0.2)
delta_p = st.sidebar.slider("Precipitation Change (%)", min_value=-40, max_value=40, value=0, step=5)

# --- Physical & Glaciological Constants (Kuhn ELA Perturbation Model) ---
base_ela = 1875.0          # Baseline equilibrium line altitude (m)
BASE_BW = 3.23              # Baseline winter balance at ELA (m ice equivalent). CALCULATED FROM 2012-2022 FIELD MEASUREMENTS AT STATION C (1900m)

DBW_DZ = 0.0055            # Winter balance vertical gradient, db_w/dz (m i.e. / m)
DR_DZ = -0.015             # Net radiation vertical gradient, dR/dz (W / m^3)
DTA_DZ = -0.0065           # Air temperature lapse rate, dT_a/dz (°C / m)

GAMMA = 12                # Turbulent heat transfer coefficient (W / (m^2 * °C))
T_MELT = 86400 * 100       # Duration of ablation season, t (s, e.g., 100 melt days)
L_FUSION = 3.34e5          # Latent heat of fusion of ice, L (J / kg)
RHO_ICE = 917.0            # Density of ice (kg / m^3) to convert kg/m^2 -> m w.e.

# Perturbations driven by sidebar sliders (plus radiation perturbation delta_R)
delta_Ta = temp - 1.6                         # Air temperature change, δT_a (°C)
delta_bw = BASE_BW * (delta_p / 100.0)     # Winter balance change, δb_w (m i.e.)
delta_R = 0.0                              # Net radiation perturbation, δR (W / m^2)

# --- Solved Equation for Δz_ela ---
# Note: (T_MELT / (L_FUSION * RHO_ICE)) converts W/m^2 (J/(s*m^2)) over season t into m i.e.
energy_to_mie = T_MELT / (L_FUSION * RHO_ICE)

numerator = energy_to_mie * (delta_R + GAMMA * delta_Ta) - delta_bw
denominator = DBW_DZ - energy_to_mie * (DR_DZ + GAMMA * DTA_DZ)

delta_ela = numerator / denominator
new_ela = base_ela + delta_ela

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
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
plt.close(fig)
buf.seek(0)

rendered_img = Image.open(buf)
w, h = rendered_img.size
trim = 50
cropped_img = rendered_img.crop((500, trim, w - trim, h - trim))
st.image(cropped_img, width=500)