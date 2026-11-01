"""Download India used-car dataset (CarDekho-style CSV)."""

from pathlib import Path
import urllib.request

URL = (
    "https://github.com/IMvision12/Car-Price_prediction/raw/main/"
    "CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cardekho.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {OUT} ...")
    urllib.request.urlretrieve(URL, OUT)
    print("Done.")


if __name__ == "__main__":
    main()
