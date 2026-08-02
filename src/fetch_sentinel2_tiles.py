import os
from dotenv import load_dotenv
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType,BBox, CRS

from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


print("CWD:", os.getcwd())
print("Script location:", __file__)
print("ID:", repr(config.sh_client_id))
print("SECRET:", repr(config.sh_client_secret))
print("sh_base_url:", config.sh_base_url)
print("sh_token_url:", config.sh_token_url)

bbox = BBox(bbox=[7.10, 4.40, 7.20, 4.50], crs=CRS.WGS84)  # Bonny area

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B03", "B04", "B08"],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B03, sample.B04, sample.B08];  // Green, Red, NIR
}
"""

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A.define_from(
            "s212a_dataspace", service_url=config.sh_base_url
        ),
        time_interval=("2025-01-01", "2025-01-31"),
    )],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=(512, 512),
    config=config,
    data_folder="./sentinel2_tiles",   # <-- add this line
)


data = request.get_data(save_data=True)

print("Fetch complete. Data length:", len(data))
print("Data folder:", config.data_folder if hasattr(config, "data_folder") else "N/A")