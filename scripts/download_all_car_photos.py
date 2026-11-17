import urllib.request
from PIL import Image
from pathlib import Path

urls = {
    "nexon_ev": "https://upload.wikimedia.org/wikipedia/commons/e/e5/2023_Tata_Nexon_XZA%2B_front_view.jpg",
    "creta": "https://upload.wikimedia.org/wikipedia/commons/b/be/HYUNDAI_CRETA_%2C_iX25_%28SU2%29_China_%281%29.jpg",
    "thar": "https://upload.wikimedia.org/wikipedia/commons/b/b6/Mahindra_Thar_Photoshoot_at_Perupalem_Beach_%28West_Godavari_District%2C_AP%2C_India%29_Djdavid.jpg",
    "hyryder": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Toyota_Urban_Cruiser_Hyryder%2C_Goa%2C_India.jpg",
    "bmw": "https://upload.wikimedia.org/wikipedia/commons/8/81/BMW_3-Series_%28G20%29_330_e_%282022%29_%2852568339927%29.jpg",
}

out_dir = Path("data/images")
out_dir.mkdir(parents=True, exist_ok=True)
headers = {'User-Agent': 'AntigravityVehicleApp/2.0 (contact@antigravity.ai)'}

for name, url in urls.items():
    dest = out_dir / f"{name}.jpg"
    print(f"Downloading {name} from {url}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        data = urllib.request.urlopen(req, timeout=20).read()
        temp_dest = out_dir / f"temp_{name}.jpg"
        temp_dest.write_bytes(data)
        
        # Resize with PIL
        with Image.open(temp_dest) as img:
            img = img.convert("RGB")
            img.thumbnail((800, 520))
            img.save(dest, "JPEG", optimize=True, quality=85)
        temp_dest.unlink(missing_ok=True)
        print(f"Saved and optimized: {dest} ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"Failed {name}: {e}")
