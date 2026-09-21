"""Guards against exactly the regression found and fixed on 2026-09-21: the
backend's category list (services/aircraft_category.CATEGORIES) went from
three to four values, and two frontend spots -- api.ts's `Categorie` type
and VueAgregee.tsx's CATEGORY_LABELS -- silently kept the old three, for
several commits, with nothing catching it (TypeScript couldn't: the type
itself was the thing that was stale, so `Record<Categorie, string>` had
nothing correct to check against).

There is no single-language type system spanning Python and TypeScript, so
this compares the four category lists as plain data instead: parses each
frontend source file's category keys back out with a small, deliberately
narrow regex (matched against a literal marker in each file, not a general
TS/JS parser) and asserts they're the exact same set as the backend's own
CATEGORIES. It's not exhaustive against every possible way these files
could be edited, but it directly re-creates the exact failure this
regression produced.
"""

import re
from pathlib import Path

from services.aircraft_category import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _extract_categorie_type_members() -> set[str]:
    text = (FRONTEND_SRC / "services" / "api.ts").read_text(encoding="utf-8")
    match = re.search(r"export type Categorie = ([^;]+);", text)
    assert match, "Categorie type not found in api.ts -- was it renamed or moved?"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def _extract_category_labels_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const CATEGORY_LABELS[^=]*=\s*\{(.*?)\};", text, re.DOTALL)
    assert match, f"CATEGORY_LABELS not found in {path}"
    return set(re.findall(r"^\s*([a-z_]+):", match.group(1), re.MULTILINE))


def test_frontend_category_type_matches_backend_categories():
    assert _extract_categorie_type_members() == set(CATEGORIES)


def test_vue_agregee_category_labels_match_backend_categories():
    path = FRONTEND_SRC / "pages" / "VueAgregee.tsx"
    assert _extract_category_labels_keys(path) == set(CATEGORIES)


def test_documentation_category_labels_match_backend_categories():
    path = FRONTEND_SRC / "pages" / "Documentation.tsx"
    assert _extract_category_labels_keys(path) == set(CATEGORIES)
