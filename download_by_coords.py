"""
download NREL NSRDB TMY data based on coordinates
use WKT POINT to set location (location_id is unnecesssary)
"""
import requests
import time
import argparse

API_KEY = "2mF0rdaGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy"
EMAIL = "magztianyi@gmail.com" #change email
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-tmy-v4-0-0-download.json"

# test: NY 5 cities (name, lat, lon)
NY_POINTS = [
    ("NewYorkCity", 40.71, -74.01),
    ("Buffalo", 42.89, -78.88),
    ("Albany", 42.65, -73.76),
    ("Syracuse", 43.05, -76.15),
    ("Rochester", 43.16, -77.61),
]


def download_by_wkt(name: str, lat: float, lon: float, year: str = "tmy-2024"):
    """
    use WKT POINT to download a specific location pt

    Args:
        name: location（display）
        lat: latitude
        lon: longitude
        year: year，default tmy-2024

    Returns:
        Download link or None
    """
    # WKT POINT: POINT(lon lat)
    wkt = f"POINT({lon} {lat})"

    params = {
        "api_key": API_KEY,
    }

    post_data = {
        "wkt": wkt,
        "names": year,
        "attributes": "air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,dhi,dni,ghi,wind_speed",
        "interval": "60",
        "utc": "false",
        "email": EMAIL,
    }

    print(f"\n[{name}] coordinate: ({lat}, {lon})")
    print(f"  WKT: {wkt}")
    print(f"  request data: {year}")

    try:
        response = requests.post(
            BASE_URL,
            params=params,
            data=post_data,
            headers={"x-api-key": API_KEY},
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            if "errors" in result and result["errors"]:
                print(f"  error: {result['errors']}")
                return None

            outputs = result.get("outputs", {})
            download_url = outputs.get("downloadUrl", "")
            message = outputs.get("message", "")

            print(f"  {message}")
            print(f"  downloard link: {download_url}")
            return download_url
        else:
            print(f"  request failed: {response.status_code}")
            print(f"  respond: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"  exception: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="download NSRDB TMY data")
    parser.add_argument("--year", default="tmy-2024", help="year (default: tmy-2024)")
    parser.add_argument("--points", default="ny", choices=["ny", "custom"],
                       help="dataset: ny=ny5city, custom=self")
    parser.add_argument("--lat", type=float, help="customlat")
    parser.add_argument("--lon", type=float, help="customlong")
    parser.add_argument("--name", default="CustomPoint", help="CustomLocationName")
    args = parser.parse_args()

    print("=" * 60)
    print("NREL NSRDB TMY data download (coordinate)")
    print("=" * 60)
    print(f"API Key: {API_KEY[:10]}...")
    print(f"Email: {EMAIL}")
    print(f"year: {args.year}")

    download_links = []

    if args.points == "custom":
        if args.lat is None or args.lon is None:
            print("error: custom requires --lat  --lon")
            return
        points = [(args.name, args.lat, args.lon)]
    else:
        points = NY_POINTS

    print(f"\n will download {len(points)} numbers of location points:")
    for name, lat, lon in points:
        print(f"  - {name}: ({lat}, {lon})")

    print("\n" + "-" * 60)

    for name, lat, lon in points:
        url = download_by_wkt(name, lat, lon, args.year)
        if url:
            download_links.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "url": url
            })
        time.sleep(2)  

    print("\n" + "=" * 60)
    print(f"Completed! {len(download_links)}/{len(points)} successful requests in total")
    print("=" * 60)

    if download_links:
        print("\ndownload links summary:")
        for item in download_links:
            print(f"\n[{item['name']}] ({item['lat']}, {item['lon']})")
            print(f"  {item['url']}")

        # save links to file
        with open("download_links.txt", "w", encoding="utf-8") as f:
            for item in download_links:
                f.write(f"{item['name']},{item['lat']},{item['lon']},{item['url']}\n")
        print("\nlink saved to download_links.txt")

        print("\nreminder: visit the link above to download ZIP files after several minutes")


if __name__ == "__main__":
    main()
