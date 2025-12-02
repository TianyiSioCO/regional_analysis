"""
batch download state NSRDB TMY data
generate grid coordinates, poll for the download link, then download and unzip
"""
import requests
import zipfile
import time
from pathlib import Path
from datetime import datetime
import json
import threading
from queue import Queue

# ============ Configuration parameters ============
API_KEY = "2mF0rdaGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy"
EMAIL = "magztianyi@gmail.com" #change email
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-tmy-v4-0-0-download.json"

# New York State boundary (simplified bounding box)
# NY_BOUNDS = {
#     "lat_min": 40.5,
#     "lat_max": 45.0,
#     "lon_min": -79.8,
#     "lon_max": -71.8
# }

MA_BOUNDS = {
    "lat_min": 41.2,
    "lat_max": 42.9,
    "lon_min": -73.5,
    "lon_max": -69.9
}


# Grid resolution (degrees) NSRDB data roughly 0.04, fix!
GRID_RESOLUTION = 0.04

# output
OUTPUT_DIR = Path(__file__).parent.parent / "AnalyzeData" / "data"

# download limit
MAX_POINTS = 2000  # maximum locations/day
REQUEST_INTERVAL = 2.1  # Request interval (s), API requires 1 request /2s
DOWNLOAD_INITIAL_WAIT = 30  # initial (s)
DOWNLOAD_RETRY_WAIT = 10  # retry (s)
MAX_DOWNLOAD_RETRIES = 60  # maximum retries
DOWNLOAD_THREADS = 3  # No of parallel download threads
# ==================================


def generate_grid_points(bounds, resolution, max_points):
    """Generate grid coordinate points"""
    points = []
    lat = bounds["lat_min"]
    while lat <= bounds["lat_max"]:
        lon = bounds["lon_min"]
        while lon <= bounds["lon_max"]:
            points.append((round(lat, 2), round(lon, 2)))
            if len(points) >= max_points:
                return points
            lon += resolution
        lat += resolution
    return points


def request_download_link(lat, lon, year="tmy-2024"):
    """Request download link for a single coordinate pt"""
    wkt = f"POINT({lon} {lat})"

    params = {"api_key": API_KEY}
    post_data = {
        "wkt": wkt,
        "names": year,
        "attributes": "air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,dhi,dni,ghi,wind_speed",
        "interval": "60",
        "utc": "false",
        "email": EMAIL,
    }

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
                return None, result["errors"]
            outputs = result.get("outputs", {})
            download_url = outputs.get("downloadUrl", "")
            return download_url, None
        else:
            return None, f"HTTP {response.status_code}"

    except Exception as e:
        return None, str(e)


def download_and_extract(item, output_dir, print_lock):
    """
    Poll, download, and unzip file in real time
    """
    lat, lon, url = item["lat"], item["lon"], item["url"]
    zip_path = output_dir / f"temp_{lat}_{lon}.zip"

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        if attempt == 1:
            wait_time = DOWNLOAD_INITIAL_WAIT
        else:
            wait_time = DOWNLOAD_RETRY_WAIT

        time.sleep(wait_time)

        try:
            response = requests.get(url, stream=True, timeout=300)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()

                if 'zip' in content_type or 'octet-stream' in content_type:
                    # download files
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # unzip
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            file_list = zip_ref.namelist()
                            zip_ref.extractall(output_dir)

                        # delete ZIP
                        zip_path.unlink()

                        csv_count = len([f for f in file_list if f.endswith('.csv')])
                        return True, f"{csv_count} numbers of CSV"

                    except zipfile.BadZipFile:
                        if zip_path.exists():
                            zip_path.unlink()
                        continue
                else:
                    if attempt % 10 == 0:
                        with print_lock:
                            print(f"      [{lat}, {lon}] Waiting for data to be ready… (retrying {attempt})")
                    continue

            elif response.status_code == 404:
                continue
            else:
                continue

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    return False, "Exceeded maximum retries"


def download_worker(download_queue, output_dir, results, print_lock, stop_event):
    """Download worker thread"""
    while not stop_event.is_set():
        try:
            item = download_queue.get(timeout=2)
            if item is None:
                break

            success, msg = download_and_extract(item, output_dir, print_lock)

            with print_lock:
                results["total"] += 1
                if success:
                    results["success"] += 1
                    print(f"  ✓ Download complete ({item['lat']}, {item['lon']}) - {msg}")
                else:
                    results["failed"] += 1
                    results["failed_items"].append(item)
                    print(f"  ✗ Download failed ({item['lat']}, {item['lon']}) - {msg}")

            download_queue.task_done()

        except Exception:
            continue


def save_progress(progress_file, results, pending_count):
    """Save progress"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({
            "success": results["success"],
            "failed": results["failed"],
            "pending": pending_count,
            "failed_items": results["failed_items"],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 70)
    print("NY state NSRDB TMY data batch download")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Maximum download pts: {MAX_POINTS}")
    print(f"Request Interval: {REQUEST_INTERVAL} seconds")
    print(f"Parallel download threads: {DOWNLOAD_THREADS}")
    print()

    # Check output directory existence
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress_file = Path(__file__).parent / "batch_progress.json"

    # Generate coordinates
    print("Generate Coordinates...")
    points = generate_grid_points(MA_BOUNDS, GRID_RESOLUTION, MAX_POINTS)
    print(f"Generate {len(points)} number of coordinates")
    print()

    # 
    # Create download queue and result stats
    download_queue = Queue()
    results = {"total": 0, "success": 0, "failed": 0, "failed_items": []}
    print_lock = threading.Lock()
    stop_event = threading.Event()

    # Start download
    download_threads = []
    for i in range(DOWNLOAD_THREADS):
        t = threading.Thread(
            target=download_worker,
            args=(download_queue, OUTPUT_DIR, results, print_lock, stop_event),
            name=f"Downloader-{i+1}"
        )
        t.daemon = True
        t.start()
        download_threads.append(t)

    # Fetch download links and put in the queue
    print("Begin fetching download links (downloads start immediately after link retrieval)...")
    print("-" * 70)

    link_success = 0
    link_failed = 0

    try:
        for i, (lat, lon) in enumerate(points):
            url, error = request_download_link(lat, lon)

            with print_lock:
                if url:
                    link_success += 1
                    print(f"[{i+1}/{len(points)}] ({lat}, {lon})")
                    print(f"    URL: {url}")

                    # Add to download queue immeidiately
                    download_queue.put({
                        "lat": lat,
                        "lon": lon,
                        "url": url
                    })
                else:
                    link_failed += 1
                    print(f"[{i+1}/{len(points)}] ({lat}, {lon}) - Fail to Fetch Link: {error}")

            # Save for every 50 itmes
            if (i + 1) % 50 == 0:
                save_progress(progress_file, results, download_queue.qsize())

            # API sleep
            time.sleep(REQUEST_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nReceive Interupt Command，Stopping...")
        stop_event.set()

    print()
    print("-" * 70)
    print(f"Fetch Link Completed: Success {link_success}, Fail {link_failed}")
    print()

    # Waiting for all downloads complete
    print("Waiting for all download tasks to finish...")
    print(f"Remaining Tasks: {download_queue.qsize()}")

    download_queue.join()

    # Stop download thread
    stop_event.set()
    for _ in download_threads:
        download_queue.put(None)
    for t in download_threads:
        t.join(timeout=5)

    # Save Progress
    save_progress(progress_file, results, 0)

    print()
    print("=" * 70)
    print("ALL COMPLETED!")
    print(f"  Fetch links: Success {link_success}, Fail {link_failed}")
    print(f"  Download files: Success {results['success']}, Fail {results['failed']}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("=" * 70)

    # count CSV files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\nDirectory contains {len(csv_files)} CSV files")

    # if failed, print
    if results["failed_items"]:
        print(f"\nFailed downloads ({len(results['failed_items'])} ):")
        for item in results["failed_items"][:10]:
            print(f"  ({item['lat']}, {item['lon']})")
        if len(results["failed_items"]) > 10:
            print(f"  ... and {len(results['failed_items']) - 10} more")


if __name__ == "__main__":
    main()
