"""Shared helpers: paths, config, geography, request cache and budget ledger."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

load_dotenv(ROOT / ".env")


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text())


SETTINGS = load_yaml("settings.yaml")
PRICING = load_yaml("pricing.yaml")


def require_env(*names: str) -> list[str]:
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")
    return [os.environ[name] for name in names]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- geography -------------------------------------------------------------

def destination_polygon():
    geojson = json.loads((ROOT / SETTINGS["destination"]["polygon_file"]).read_text())
    return shape(geojson["features"][0]["geometry"])


def in_destination(lat: float, lon: float, polygon=None) -> bool:
    polygon = polygon or destination_polygon()
    return polygon.contains(Point(lon, lat))


def destination_places():
    """The analysis population: accommodation inside the destination polygon.

    Every metric must take its denominator from here, never filter places.parquet itself;
    out-of-destination places stay in the file only so AI mentions of them can be matched.
    """
    import pandas as pd

    places = pd.read_parquet(PROCESSED / "places.parquet")
    return places[places["in_destination"] & places["is_accommodation"]].reset_index(drop=True)


def check_population(place_uids, context: str) -> None:
    """Fail loudly if a metric's place set differs from destination_places()."""
    expected = set(destination_places()["place_uid"])
    actual = set(place_uids)
    if actual != expected:
        raise AssertionError(f"{context}: {len(actual - expected)} places outside the destination population, "
                             f"{len(expected - actual)} destination places missing.")


def haversine_km(lat_a, lon_a, lat_b, lon_b) -> float:
    earth_radius_km = 6371.0
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    delta_phi, delta_lambda = phi_b - phi_a, math.radians(lon_b - lon_a)
    chord = math.sin(delta_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(chord))


def km_from_center(lat: float, lon: float) -> float:
    center_lat, center_lon = SETTINGS["destination"]["center"]
    return haversine_km(lat, lon, center_lat, center_lon)


# --- cache -----------------------------------------------------------------

def request_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def cache_get(folder: Path, key: str):
    path = folder / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cache_put(folder: Path, key: str, record: dict) -> None:
    """Atomic write: a process killed mid-write leaves the old file or none, never a broken one."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{key}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    os.replace(temporary, folder / f"{key}.json")


# --- budget ----------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Running spend estimate shared by every stage, persisted as a JSONL ledger."""

    def __init__(self):
        config = SETTINGS["budget"]
        self.warn, self.cap = config["warn_usd"], config["hard_cap_usd"]
        self.ledger = ROOT / config["ledger_file"]

    def spent(self) -> float:
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.ledger.read_text().splitlines() if line)

    def check(self, next_cost: float) -> None:
        if self.spent() + next_cost > self.cap:
            raise BudgetExceeded(f"Hard cap ${self.cap} would be exceeded (spent ${self.spent():.2f}).")

    def record(self, stage: str, provider: str, usd: float, detail: str = "", **extra) -> None:
        before = self.spent()
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as ledger:
            ledger.write(json.dumps({"timestamp": now_iso(), "stage": stage, "provider": provider,
                                     "usd": usd, "detail": detail, **extra}) + "\n")
        if before < self.warn <= before + usd:
            print(f"WARNING: spend passed the ${self.warn} target (now ${before + usd:.2f}).")
