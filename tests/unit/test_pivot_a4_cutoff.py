"""Vérifie que le script de replay ne lit jamais ≥ 2024."""
import re
from pathlib import Path


def test_no_2024_2025_2026_literals() -> None:
    """Le script ne doit pas contenir de littéraux d'années test set."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    # Exclure les chaînes en commentaires
    code_lines = [
        line for line in script.splitlines()
        if not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    forbidden = re.findall(r'"(2024|2025|2026)', code)
    assert not forbidden, f"Test set literals found: {forbidden}"


def test_cutoff_constant_present() -> None:
    """Le script doit définir et utiliser CUTOFF_DATE."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    assert "CUTOFF_DATE" in script, "CUTOFF_DATE manquant"
    assert "2023-12-31" in script, "Date cutoff 2023-12-31 manquante"


def test_cutoff_used_at_least_twice() -> None:
    """CUTOFF_DATE doit apparaître au moins 2 fois (définition + utilisation)."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    count = script.count("CUTOFF_DATE")
    assert count >= 2, f"CUTOFF_DATE apparaît seulement {count} fois, ≥2 attendu"


def test_no_read_oos_call() -> None:
    """Le script ne doit pas appeler read_oos()."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    assert "read_oos" not in script, "read_oos() trouvé — interdit en A4"


def test_no_test_set_lock_import() -> None:
    """Le script ne doit pas importer TEST_SET_LOCK."""
    script = Path("scripts/run_pivot_a4_replay.py").read_text(encoding="utf-8")
    assert "TEST_SET_LOCK" not in script, "TEST_SET_LOCK importé — interdit en A4"
