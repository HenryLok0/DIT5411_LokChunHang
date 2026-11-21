"""Download and verify the original 12-part assignment dataset on Windows or Linux."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen

REVISION = "2388ea0be27a3b966994c094e0b7fff0357deea9"
BASE_URL = f"https://raw.githubusercontent.com/chenkenanalytic/handwritting_data_all/{REVISION}"
SHA256 = "74607b8d642a4dfd640586efd221cd935bdf16c93c83f2eb9e5092befde60420"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(dest):
    if dest.exists():
        if checksum(dest) != SHA256:
            raise ValueError(f"{dest} already exists but isn't the expected archive.")
        print(f"Verified existing {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dit5411-parts-", dir=dest.parent) as local:
        local = Path(local)

        def get_part(i):
            name = f"all_data.zip.{i:03d}"
            path = local / name
            expected = 99_614_720 if i < 12 else 44_338_542
            for attempt in range(3):
                try:
                    with urlopen(f"{BASE_URL}/{name}", timeout=60) as reply, path.open("wb") as out:
                        shutil.copyfileobj(reply, out, length=1024 * 1024)
                    if path.stat().st_size != expected:
                        raise OSError(f"{name} was not downloaded in full.")
                    print(f"Downloaded {name}", flush=True)
                    return path
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        with ThreadPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(get_part, range(1, 13)))
        combined = local / "all_data.zip"
        with combined.open("wb") as out:
            for path in parts:
                with path.open("rb") as stream:
                    shutil.copyfileobj(stream, out, length=1024 * 1024)
                path.unlink()
        if checksum(combined) != SHA256:
            raise ValueError("Combined ZIP does not match the published input used by this project.")
        combined.replace(dest)
    print(f"Verified {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/all_data.zip"))
    download(parser.parse_args().output)
