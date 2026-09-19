"""Regenerates data/reference/aircraft_categories.csv (typecode -> categorie),
the table that replaces "OpenAP recognizes this typecode" as the
categorization boundary (services/aircraft_category.py).

Why this change: OpenAP only recognizes around forty typecodes (its own
simulation list) — using that as a proxy for "what kind of aircraft is this"
rather than "can OpenAP simulate it" let real airliners (A330-900/A339,
A220/BCS1/BCS3, CRJ-1000/CRJX, Dash 8-400/DH8D, E195-E2/E295, 787-10/B78X,
767-400/B764...) fall into "petit_avion" simply because OpenAP doesn't have
their performance sheet yet — and conversely, two business jets (c550,
glf6) that OpenAP natively lists ended up classified as "avion_ligne". Found
directly on the training dataset (see the conversation history), not a
hypothesis.

Built in layers, each independently verifiable:
  1. helicopter: the `icaoaircrafttype` field of aircraft_database.csv (H =
     rotorcraft per the ICAO Doc 8643 designation), aggregated by typecode
     (not by icao24 — much denser, and a real typecode has no reason to have
     an inconsistent icaoaircrafttype from one airframe to another). 100%
     reliable on the typecodes checked by hand.
  2. avion_ligne (automatic): natively recognized by OpenAP, except for
     c550/glf6 (and any typecode resolved to them via synonym) — the only
     two business jets on OpenAP's list, reclassified as jet_affaire below.
  3. Everything else (`MANUAL_OVERRIDES`): manual classification, verified
     by hand, of typecodes OpenAP doesn't recognize and that aren't
     helicopters — only for typecodes actually encountered in
     data/processed/*.parquet and data/reference/flight_pool.jsonl (a
     closed, checkable list, not an attempt to cover every aircraft in the
     world). A typecode absent from all three layers stays "unidentified" —
     no score is computed for it (pipeline.py) rather than a guessed
     default classification.

Usage: python scripts/build_aircraft_categories.py
"""

import sys
from pathlib import Path

import glob
import os

import pandas as pd
from openap import prop

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.aircraft_database import _get_aircraft_db  # noqa: E402

_OPENAP_AIRCRAFT_DIR = os.path.dirname(prop.__file__) + "/data/aircraft/"

OUTPUT_PATH = BACKEND_DIR.parent / "data" / "reference" / "aircraft_categories.csv"

# The only two business jets OpenAP's native list contains — any typecode
# resolved to them via synonym (cf. OpenAP's own _synonym.csv:
# c25a/c525/c56x/pc24 -> c550, glf5/gl5t/lj45 -> glf6) inherits the same
# reclassification, for the same reason: neither is an airliner.
_OPENAP_BUSINESS_JETS = {"c550", "glf6"}


def _openap_resolved_ac(typecode: str) -> str | None:
    """The OpenAP performance file actually used for this typecode, after
    synonym resolution (FlightGenerator.ac keeps the ORIGINAL requested
    name, not the file actually loaded — the same resolution prop.aircraft()
    does needs to be replayed to know which one), or None if OpenAP doesn't
    recognize it at all."""
    ac = typecode.lower()
    if glob.glob(_OPENAP_AIRCRAFT_DIR + ac + ".yml"):
        return ac
    syno = prop.aircraft_synonym.query("orig==@ac")
    if syno.shape[0] > 0:
        new_ac = syno.new.iloc[0]
        if glob.glob(_OPENAP_AIRCRAFT_DIR + new_ac + ".yml"):
            return new_ac
    return None


# Seen in data/processed/*.parquet and data/reference/flight_pool.jsonl, not
# recognized by OpenAP, not helicopters — classified by hand (looking up the
# real model by ICAO typecode). "737"/"C17"/"C30J": an ambiguous typecode or
# a military transport, assigned to avion_ligne by size/performance rather
# than perfect taxonomic accuracy (none of the 4 categories fits them
# exactly).
MANUAL_OVERRIDES: dict[str, str] = {
    # --- avion_ligne: real airliners/regional aircraft, absent from OpenAP ---
    "737": "avion_ligne", "A337": "avion_ligne", "A338": "avion_ligne", "A339": "avion_ligne",
    "A346": "avion_ligne", "A35K": "avion_ligne", "A3ST": "avion_ligne", "A400": "avion_ligne",
    "AN26": "avion_ligne", "AT43": "avion_ligne", "AT45": "avion_ligne", "AT73": "avion_ligne",
    "B712": "avion_ligne", "B736": "avion_ligne", "B753": "avion_ligne", "B764": "avion_ligne",
    "B78X": "avion_ligne", "BCS1": "avion_ligne", "BCS3": "avion_ligne", "BE30": "avion_ligne",
    "C130": "avion_ligne", "C17": "avion_ligne", "C295": "avion_ligne", "C30J": "avion_ligne",
    "CRJ7": "avion_ligne", "CRJX": "avion_ligne", "D228": "avion_ligne", "D328": "avion_ligne",
    "DH8A": "avion_ligne", "DH8B": "avion_ligne", "DH8C": "avion_ligne", "DH8D": "avion_ligne",
    "E120": "avion_ligne", "E135": "avion_ligne", "E295": "avion_ligne", "E45X": "avion_ligne",
    "E75S": "avion_ligne", "F100": "avion_ligne", "IL62": "avion_ligne", "J328": "avion_ligne",
    "L410": "avion_ligne", "SF34": "avion_ligne", "SW4": "avion_ligne",
    # --- jet_affaire: business/corporate jets and turboprops ---
    "B350": "jet_affaire", "BE10": "jet_affaire", "BE20": "jet_affaire", "BE40": "jet_affaire",
    "BE9L": "jet_affaire", "C25B": "jet_affaire", "C25C": "jet_affaire", "C25M": "jet_affaire",
    "C425": "jet_affaire", "C441": "jet_affaire", "C510": "jet_affaire", "C551": "jet_affaire",
    "C55B": "jet_affaire", "C560": "jet_affaire", "C650": "jet_affaire", "C680": "jet_affaire",
    "C68A": "jet_affaire", "C700": "jet_affaire", "C750": "jet_affaire", "CL30": "jet_affaire",
    "CL35": "jet_affaire", "CL60": "jet_affaire", "E35L": "jet_affaire", "E50P": "jet_affaire",
    "E545": "jet_affaire", "E550": "jet_affaire", "E55P": "jet_affaire", "EA50": "jet_affaire",
    "F2TH": "jet_affaire", "F900": "jet_affaire", "FA10": "jet_affaire", "FA20": "jet_affaire",
    "FA50": "jet_affaire", "FA6X": "jet_affaire", "FA7X": "jet_affaire", "FA8X": "jet_affaire",
    "G200": "jet_affaire", "G280": "jet_affaire", "G450": "jet_affaire", "GA5C": "jet_affaire",
    "GALX": "jet_affaire", "GL7T": "jet_affaire", "GLEX": "jet_affaire", "GLF4": "jet_affaire",
    "H25B": "jet_affaire", "H25C": "jet_affaire", "H750": "jet_affaire", "H900": "jet_affaire",
    "HA4T": "jet_affaire", "HDJT": "jet_affaire", "LJ31": "jet_affaire", "LJ35": "jet_affaire",
    "LJ40": "jet_affaire", "LJ60": "jet_affaire", "LJ75": "jet_affaire", "P180": "jet_affaire",
    "PRM1": "jet_affaire", "SF50": "jet_affaire", "T38": "jet_affaire",
    # --- petit_avion: light aviation (flight school, touring, ULM, propeller) ---
    "AA5": "petit_avion", "AAT3": "petit_avion", "AP32": "petit_avion", "B36T": "petit_avion",
    "BE23": "petit_avion", "BE33": "petit_avion", "BE35": "petit_avion", "BE58": "petit_avion",
    "BE65": "petit_avion", "BE95": "petit_avion", "C150": "petit_avion", "C152": "petit_avion",
    "C172": "petit_avion", "C182": "petit_avion", "C208": "petit_avion", "C340": "petit_avion",
    "C404": "petit_avion", "C42": "petit_avion", "C72R": "petit_avion", "C82S": "petit_avion",
    "C82T": "petit_avion", "CLON": "petit_avion", "DA40": "petit_avion", "DA42": "petit_avion",
    "DA62": "petit_avion", "DHC2": "petit_avion", "DHC6": "petit_avion", "DR40": "petit_avion",
    "DV20": "petit_avion", "EFOX": "petit_avion", "HR20": "petit_avion", "LA4": "petit_avion",
    "M20P": "petit_avion", "M20T": "petit_avion", "N3N": "petit_avion", "P208": "petit_avion",
    "P28A": "petit_avion", "P28B": "petit_avion", "P28U": "petit_avion", "P32R": "petit_avion",
    "P46T": "petit_avion", "P68": "petit_avion", "PA31": "petit_avion", "PA32": "petit_avion",
    "PA34": "petit_avion", "PA38": "petit_avion", "PA44": "petit_avion", "PA46": "petit_avion",
    "PC12": "petit_avion", "PC9": "petit_avion", "RV14": "petit_avion", "RV9": "petit_avion",
    "S22T": "petit_avion", "SR20": "petit_avion", "SR22": "petit_avion", "SS2T": "petit_avion",
    "ST75": "petit_avion", "T206": "petit_avion", "T210": "petit_avion", "T34P": "petit_avion",
    "TBM7": "petit_avion", "TBM8": "petit_avion", "TBM9": "petit_avion", "TEX2": "petit_avion",
    "Y18T": "petit_avion",
}


def build() -> pd.DataFrame:
    db = _get_aircraft_db()
    icaotype_by_typecode = (
        db.dropna(subset=["typecode"])
        .groupby("typecode")["icaoaircrafttype"]
        .agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().shape[0] > 0 else None)
    )

    rows = []
    for typecode, icaotype in icaotype_by_typecode.items():
        typecode_upper = typecode.upper()
        if isinstance(icaotype, str) and icaotype.startswith("H"):
            categorie = "helicoptere"
        else:
            resolved = _openap_resolved_ac(typecode)
            if resolved is not None:
                categorie = "jet_affaire" if resolved in _OPENAP_BUSINESS_JETS else "avion_ligne"
            else:
                categorie = MANUAL_OVERRIDES.get(typecode_upper)
        if categorie is not None:
            rows.append({"typecode": typecode_upper, "categorie": categorie})

    out = pd.DataFrame(rows).drop_duplicates(subset=["typecode"]).sort_values("typecode")
    return out


def main() -> None:
    out = build()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"{OUTPUT_PATH}: {len(out)} typecodes classified")
    print(out["categorie"].value_counts())


if __name__ == "__main__":
    main()
