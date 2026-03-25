# %% [markdown]
# # HISDAC-US Version II Downloader
# ### Author: D. Acosta-Reyes
# ### Date: 2026-03-25
# ### Supervisor: Dr. J. Wartman
#
# This script allows you to download HISDAC-US datasets from Harvard Dataverse.
# 
# Run each cell with Shift+Enter in VS Code.
#
# Features:
# - Choose 1 of 7 HISDAC-US datasets
# - List files available in the selected Dataverse record
# - Download all files or selected files to a local folder

# %%
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DATASETS: dict[int, dict[str, str]] = {
	1: {
		"name": "Historical Land Use 1940-2020: Class Counts V2",
		"doi": "doi:10.7910/DVN/PRJBUF",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PRJBUF#",
	},
	2: {
		"name": "Historical Land Use 1940-2020: Major Class V2",
		"doi": "doi:10.7910/DVN/CHLNQG",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CHLNQG#",
	},
	3: {
		"name": "Historical Built-up Records (BUPR) V2",
		"doi": "doi:10.7910/DVN/45B8IU",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/45B8IU#",
	},
	4: {
		"name": "Historical Built-up Property Locations (BUPL) V2",
		"doi": "doi:10.7910/DVN/U2P66Z",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/U2P66Z#",
	},
	5: {
		"name": "Historical Built-up Areas (BUA) V2",
		"doi": "doi:10.7910/DVN/JHY9BT",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JHY9BT#",
	},
	6: {
		"name": "Historical Built-up Intensity Layer (BUI) V2",
		"doi": "doi:10.7910/DVN/CSLOJP",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CSLOJP#",
	},
	7: {
		"name": "Historical Settlement Year Built Layer 1810-2020 V2",
		"doi": "doi:10.7910/DVN/HHFM5E",
		"url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HHFM5E#",
	},
}

API_ROOT = "https://dataverse.harvard.edu/api"
USER_AGENT = "Mozilla/5.0 (HISDAC-US-downloader)"
DATA_PATH = Path("/Users/danielacosta/Library/CloudStorage/OneDrive-UW/0 - DA General Exam/Paper 2 - Temporal Dynamics/Data")
DOWNLOAD_ROOT = DATA_PATH / "HISDAC_US_V2"


def _http_get_json(url: str) -> dict[str, Any]:
	"""Fetch JSON data from a URL and return it as a dictionary."""
	req = request.Request(url, headers={"User-Agent": USER_AGENT})
	with request.urlopen(req) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _human_size(n_bytes: int) -> str:
	"""Convert a byte count to a readable size string."""
	step = 1024.0
	units = ["B", "KB", "MB", "GB", "TB"]
	size = float(n_bytes)
	for unit in units:
		if size < step or unit == units[-1]:
			return f"{size:.1f} {unit}"
		size /= step
	return f"{n_bytes} B"


def _safe_name(text: str) -> str:
	"""Make a folder-safe name by replacing special characters."""
	return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def fetch_dataset_files(doi: str) -> list[dict[str, Any]]:
	"""Get the list of files available for a dataset DOI."""
	url = f"{API_ROOT}/datasets/:persistentId/?{parse.urlencode({'persistentId': doi})}"
	payload = _http_get_json(url)
	version = payload["data"]["latestVersion"]
	files = version.get("files", [])

	result: list[dict[str, Any]] = []
	for f in files:
		datafile = f.get("dataFile", {})
		result.append(
			{
				"label": f.get("label") or datafile.get("filename", "unknown"),
				"filename": datafile.get("filename", "unknown"),
				"id": datafile.get("id"),
				"size": datafile.get("filesize", 0),
				"restricted": bool(f.get("restricted", False)),
				"content_type": datafile.get("contentType", "unknown"),
			}
		)
	return result


def download_datafile(datafile_id: int, destination: Path, overwrite: bool = False) -> None:
	"""Download one Dataverse file to the target path."""
	if destination.exists() and not overwrite:
		print(f"Skip existing file: {destination.name}")
		return

	url = f"{API_ROOT}/access/datafile/{datafile_id}"
	req = request.Request(url, headers={"User-Agent": USER_AGENT})

	with request.urlopen(req) as resp, destination.open("wb") as f:
		total = int(resp.headers.get("Content-Length", "0"))
		downloaded = 0
		chunk_size = 1024 * 1024

		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			f.write(chunk)
			downloaded += len(chunk)

			if total > 0:
				pct = (downloaded / total) * 100
				print(f"  {destination.name}: {pct:6.2f}% ({_human_size(downloaded)} / {_human_size(total)})", end="\r")

	if total > 0:
		print(" " * 120, end="\r")
	print(f"Downloaded: {destination}")


def choose_dataset() -> dict[str, str]:
	"""Show dataset options and return the selected dataset."""
	print("\nAvailable HISDAC-US datasets:\n")
	for idx, info in DATASETS.items():
		print(f"{idx}. {info['name']}")
		print(f"   DOI: {info['doi']}")

	while True:
		choice = input("\nChoose dataset number (1-7): ").strip()
		if choice.isdigit() and int(choice) in DATASETS:
			return DATASETS[int(choice)]
		print("Invalid selection. Enter a number from 1 to 7.")


def choose_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Show files and return the user's selected file list."""
	if not files:
		return []

	print("\nFiles in dataset:\n")
	for i, f in enumerate(files, start=1):
		lock = " [restricted]" if f["restricted"] else ""
		print(f"{i:>3}. {f['filename']} ({_human_size(f['size'])}){lock}")

	print("\nType 'all' to download all files, or comma-separated indices (for example: 1,3,5)")
	while True:
		raw = input("File selection: ").strip().lower()
		if raw == "all":
			return files

		selected: list[dict[str, Any]] = []
		try:
			indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
			if not indices:
				raise ValueError
			for idx in indices:
				if idx < 1 or idx > len(files):
					raise IndexError
				selected.append(files[idx - 1])
			return selected
		except (ValueError, IndexError):
			print("Invalid selection. Use 'all' or comma-separated valid indices.")


# %% [markdown]
# ## Run The Downloader
#
# This cell will:
# 1. Ask you which HISDAC-US dataset to use.
# 2. Query Dataverse and list files for that dataset.
# 3. Ask which files to download.
# 4. Save files under DATA_PATH/HISDAC_US_V2/<dataset_name>/

# %%
# Main run flow: choose dataset, choose files, download, and handle errors.
try:
	dataset = choose_dataset()
	print(f"\nSelected: {dataset['name']}")
	print(f"Dataset page: {dataset['url']}")

	all_files = fetch_dataset_files(dataset["doi"])
	selected_files = choose_files(all_files)

	if not selected_files:
		print("No files selected. Exiting.")
	else:
		out_dir = DOWNLOAD_ROOT / _safe_name(dataset["name"])
		out_dir.mkdir(parents=True, exist_ok=True)

		print(f"\nDownloading {len(selected_files)} file(s) to: {out_dir.resolve()}\n")
		for f in selected_files:
			if f["id"] is None:
				print(f"Skipping {f['filename']} (missing datafile id)")
				continue
			destination = out_dir / f["filename"]
			download_datafile(int(f["id"]), destination)

		print("\nDone.")

except error.HTTPError as e:
	if e.code in (401, 403):
		print("Access denied by Dataverse. Some files may require an access request or login.")
	else:
		print(f"HTTP error: {e.code} {e.reason}")
except error.URLError as e:
	print(f"Network error: {e.reason}")
except KeyboardInterrupt:
	print("\nCancelled by user.")

# %%
