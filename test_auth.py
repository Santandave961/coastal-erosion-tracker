"""
Direct test of Sentinel Hub OAuth credentials, bypassing the sentinelhub
library's retry/session wrapping so we see the REAL error from the server.

Run: python test_auth.py
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

client_id = os.getenv("SH_CLIENT_ID")
client_secret = os.getenv("SH_CLIENT_SECRET")

print("Client ID:", repr(client_id))
print("Client secret:", repr(client_secret))

if not client_id or not client_secret:
    print("\nERROR: credentials not loaded from .env — fix that first.")
    exit(1)

token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

response = requests.post(
    token_url,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
)

print("\nStatus code:", response.status_code)
print("Response body:", response.text)