import xarray as xr
import geopandas as gpd
import numpy as np
import glob
import json
from datetime import datetime
from wrf import getvar, ALL_TIMES
from shapely.geometry import box
from netCDF4 import Dataset
from shapely.vectorized import contains
from shapely.geometry import Point
import base64
from PIL import Image
import io
import rasterio
import os

# Caminho para o TIFF do Copernicus (imagem completa)
#Pdash=os.getenv("Pdash") # pro pampa
Pdash="/home/user/Build_WRF/dashboard" # pro user
SAT_TIFF = f"{Pdash}/satelite/basemap_zoom14_EPSG4326.tif" # basemap_zoom14.tif #candiota.tiff"
SAT_PNG  = f"{Pdash}/satelite/sentinel_recorte.png"


# -------------------------------
#  FUNÇÕES PARA SATÉLITE
# -------------------------------
def recortar_tiff_para_png(tiff_path, lon_min, lat_min, lon_max, lat_max, out_png):
    """
    Recorta o TIFF original Copernicus preservando as cores reais,
    fazendo normalização correta caso os valores não estejam em 0–255.
    """

    with rasterio.open(tiff_path) as src:
        window = rasterio.windows.from_bounds(
            lon_min, lat_min, lon_max, lat_max, src.transform
        )
        recorte = src.read(window=window)  # [bands, H, W]

        # --- Detectar tipo de dado ---
        dtype = recorte.dtype

        # Se for float → normalizar 0-1 para 0-255
        if np.issubdtype(dtype, np.floating):
            recorte = np.clip(recorte, 0, 1)  # caso venha em reflectância
            recorte = (recorte * 255).astype(np.uint8)

        # Se for uint16 → normalizar para 0–255
        elif dtype == np.uint16:
            recorte = (recorte / 256).astype(np.uint8)

        # Converter [bands, H, W] -> [H, W, bands]
        img = np.moveaxis(recorte, 0, -1)
        img = img[::4, ::4]

        # Salvar como PNG
        Image.fromarray(img).save(out_png, format="PNG")

        return out_png

def load_satellite_as_base64(path):
    img = Image.open(path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


# -------------------------------
#  MARCADORES
# -------------------------------
MARKERS = [
    {"nome": "Usina", "lon": -53.775394, "lat": -31.450326},
    {"nome": "Estação", "lon": -53.747327778, "lat": -31.45132222}
]

gdf_markers = gpd.GeoDataFrame(
    MARKERS,
    geometry=[Point(m["lon"], m["lat"]) for m in MARKERS],
    crs="EPSG:4326"
)

hoje = datetime.now()
data_str = hoje.strftime("%d%m%y")
#BASE_DIR = "/home/lmqa/OPERACAO/dashboard/"+data_str # pro pampa
BASE_DIR = "/home/user/Build_WRF/dashboard/"+data_str # pro user
os.makedirs(BASE_DIR, exist_ok=True)

#PWRF=os.getenv("PWRF") # pro pampa
#Putils=os.getenv("Putils") # pro pampa
PWRF="/home/user/Build_WRF/dashboard/wrfout" # pro user
Putils="/home/user/utils" # pro user

PADRAO_WRF = f"{PWRF}/wrfout_d03*" 
SHAPE = f"{Putils}/base_cartografica/RS_Municipios_2024/RS_Municipios_2024.shp"

# Camadas extras
SHAPE_RODOVIAS = f"{Putils}/base_cartografica/Sistema_Transporte/Trecho_Rodoviario.shp"
SHAPE_FERROVIAS = f"{Putils}/base_cartografica/Sistema_Transporte/Trecho_Ferroviario.shp"
SHAPE_AREA_EDIFICADA = f"{Putils}/base_cartografica/Localidades/Area_Edificada.shp"


VAR_LIST = {
    "T2": {"src": "ds", "convert": "K_to_C", "units": "°C"},
    "rh2": {"src": "wrf", "name": "rh2", "units": "%"},
    "WSPD10": {"src": "wrf", "name": "wspd_wdir10", "index": 0, "units": "m/s"},
    "WDIR10": {"src": "wrf", "name": "wspd_wdir10", "index": 1, "units": "°"},

    # QA
    "so2": {"src": "ds", "units": "ppmv"},
    "no": {"src": "ds", "units": "ppmv"},
    "P10": {"src": "ds", "units": "µg/m³"},
}

NIVEL = 0

print("Abrindo wrfout...")
arquivos = sorted(glob.glob(PADRAO_WRF))
wrf_files = [Dataset(f) for f in arquivos]

ds = xr.open_mfdataset(
    arquivos,
    concat_dim="Time",
    combine="nested",
    engine="netcdf4",
    parallel=False
)

lat = ds["XLAT"].isel(Time=0).values
lon = ds["XLONG"].isel(Time=0).values

STEP = 4  # testa 2, 3, 4... (quanto maior, mais leve)

lat = lat[::STEP, ::STEP]
lon = lon[::STEP, ::STEP]

# EXTENSÃO DO DOMÍNIO
lon_min, lon_max = float(lon.min()), float(lon.max())
lat_min, lat_max = float(lat.min()), float(lat.max())
bbox = box(lon_min, lat_min, lon_max, lat_max)

print("Lendo shapefile principal...")
shape = gpd.read_file(SHAPE).to_crs("EPSG:4326")
shape_clip = shape.clip(gpd.GeoSeries([bbox], crs="EPSG:4326"))


# -------------------------------
# CAMADAS EXTRAS
# -------------------------------
print("Lendo camadas adicionais...")

gdf_rodovias = gpd.read_file(SHAPE_RODOVIAS).to_crs("EPSG:4326")
gdf_ferrovias = gpd.read_file(SHAPE_FERROVIAS).to_crs("EPSG:4326")
gdf_area_edificada = gpd.read_file(SHAPE_AREA_EDIFICADA).to_crs("EPSG:4326")

gdf_bbox = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

rodovias_clip = gpd.clip(gdf_rodovias, gdf_bbox)
ferrovias_clip = gpd.clip(gdf_ferrovias, gdf_bbox)
area_edificada_clip = gpd.clip(gdf_area_edificada, gdf_bbox)


# -------------------------------
# RECORTE DO SATÉLITE
# -------------------------------
print("Recortando imagem de satélite (Copernicus)...")
sat_png_path = recortar_tiff_para_png(
    SAT_TIFF, lon_min, lat_min, lon_max, lat_max, SAT_PNG
)
sat_base64 = load_satellite_as_base64(sat_png_path)


# -------------------------------
# JSON BASE
# -------------------------------
grid_data = {
    "lon": lon.tolist(),
    "lat": lat.tolist(),
    "time": ds["XTIME"].astype(str).values.tolist()
}

layers_data = {
    "geojson": json.loads(shape_clip.to_json()),
    "geojson_layers": {
        "rodovias": json.loads(rodovias_clip.to_json()),
        "ferrovias": json.loads(ferrovias_clip.to_json()),
        "area_edificada": json.loads(area_edificada_clip.to_json()),
        "marcadores": json.loads(gdf_markers.to_json())
    },
    "background_layers": {
        "satellite": {
            "type": "image",
            "format": "png",
            "base64": sat_base64,
            "bbox": [lon_min, lat_min, lon_max, lat_max]
        }
    }
}
with open(f"{BASE_DIR}/grid.json", "w") as f:
    json.dump(grid_data, f)

with open(f"{BASE_DIR}/layers.json", "w") as f:
    json.dump(layers_data, f)


# -------------------------------
# CHUVA
# -------------------------------
print("Calculando chuva...")

rainnc = getvar(wrf_files, "RAINNC", timeidx=ALL_TIMES)
rainc  = getvar(wrf_files, "RAINC", timeidx=ALL_TIMES)

RAIN_ACC = rainnc + rainc

time_vals = ds["XTIME"].values.astype("datetime64[m]")

dt_hours = xr.DataArray(
    np.diff(time_vals) / np.timedelta64(1, "h"),
    coords={"Time": RAIN_ACC.Time[1:]},
    dims=["Time"]
)

RAINRATE = RAIN_ACC.diff("Time") / dt_hours


def save_rain_var(name, data_array, unit):

    arr = data_array.values[:, ::STEP, ::STEP].astype(float)

    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        print(f"⚠️ {name} sem dados válidos")
        return

    var_dir = f"{BASE_DIR}/{name}"
    os.makedirs(var_dir, exist_ok=True)
    
    arr_clean = np.where(np.isfinite(arr), arr, np.nan)

    data_all = np.round(arr_clean, 2)

    for t in range(0, data_all.shape[0], TIME_STEP):

        timestep_data = {
            "data": data_all[t].tolist(),
            "meta": {
                "zmin": float(vals.min()),
                "zmax": float(vals.max()),
                "units": unit
            }
        }

        with open(f"{var_dir}/t{t:02d}.json", "w") as f:
            json.dump(timestep_data, f)

TIME_STEP = 3  # pega 1 a cada 3 horas
save_rain_var("RAIN_ACC", RAIN_ACC, "mm")
save_rain_var("RAINRATE", RAINRATE, "mm/h")

# --------------------------------------------
# VARIÁVEIS DO WRF + CONVERSÃO QUÍMICA
# --------------------------------------------
print("Processando variáveis...")

# Massa molar para conversão ppmv → µg/m³
MOLAR_MASS = {
    "so2": 64.066,
    "no": 30.006,
}

# Temperatura e pressão em todas as horas
T_K = ds["T2"].values
P_pa = ds["PSFC"].values


for var, cfg in VAR_LIST.items():
    print(f"  → {var}")

    try:
        # Leitura da variável
        if cfg["src"] == "wrf":
            v = getvar(wrf_files, cfg["name"], timeidx=ALL_TIMES)

            if cfg["name"] == "wspd_wdir10" and "index" in cfg:
                v = v.isel(wspd_wdir=cfg["index"])

            if cfg["name"] == "cape_2d" and "index" in cfg:
                v = v.isel(mcape_mcin_lcl_lfc=cfg["index"])

        else:
            if var not in ds:
                print(f"⚠️ {var} não existe no wrfout")
                continue
            v = ds[var]

        # Nível vertical (se existir)
        if "bottom_top" in v.dims:
            v = v.isel(bottom_top=0)

        # Kelvin → Celsius
        if cfg.get("convert") == "K_to_C":
            v = v - 273.15

        # Conversão ppmv → µg/m³
        if cfg["units"] in ["ppm", "ppmv"] and var in MOLAR_MASS:
            M = MOLAR_MASS[var]
            print(f"    Convertendo {var} ppmv → µg/m³ (M={M})")

            R = 8.314462618

            mixing_ratio = v * 1e-6
            v = (mixing_ratio * M * P_pa / (R * T_K)) * 1e6

            cfg["units"] = "µg/m³"

        # Sanitizar NaNs para JSON
        arr = v.values[:, ::STEP, ::STEP].astype(np.float32)
        arr_clean = np.where(
			(np.isfinite(arr)) & (arr < 1e10) & (arr > -1e10),
			arr,
			np.nan
		)
        vals = arr[np.isfinite(arr)]

        if vals.size == 0:
            print(f"⚠️ {var} sem dados válidos")
            continue

        zmin = float(vals.min())
        zmax = float(vals.max())

        # Salvar no JSON
        var_dir = f"{BASE_DIR}/{var}"
        os.makedirs(var_dir, exist_ok=True)

        data_all = np.round(arr_clean, 1)

        for t in range(data_all.shape[0]):

            if var == "P10":
                max_val = np.nanmax(data_all[t])
                print(f"P10 | t={t:02d} | max = {max_val:.2f} µg/m³")

            timestep_data = {
                "data": data_all[t].tolist(),
                "meta": {
                    "zmin": zmin,
                    "zmax": zmax,
                    "units": cfg["units"]
                }
            }

            with open(f"{var_dir}/t{t:02d}.json", "w") as f:
                json.dump(timestep_data, f, separators=(",", ":"))

    except Exception as e:
        print(f"❌ Erro em {var}: {e}")

print("✅ Dados exportados em estrutura otimizada")
print(f"Pasta criada: {BASE_DIR}")
