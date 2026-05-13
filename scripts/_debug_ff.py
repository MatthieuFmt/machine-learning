"""Debug v2: inspecte les cellules individuelles pour impact + actual/forecast/previous."""
import cloudscraper
from bs4 import BeautifulSoup

s = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "mobile": False})
r = s.get("https://www.forexfactory.com/calendar?week=jan8.2024", timeout=30)
soup = BeautifulSoup(r.text, "html.parser")

table = soup.select_one("table.calendar__table")
rows = table.find_all("tr")

for j, row in enumerate(rows[:20]):
    cls = row.get("class", [])
    if "calendar__row--day-breaker" in cls:
        continue
    cells = row.find_all("td")
    if len(cells) < 8:
        continue

    # Afficher TOUTES les cellules avec leur HTML pour les lignes 4-7
    if j in [4, 5, 6]:
        print(f"\n--- Row {j} (class={cls}) ---")
        for ci, c in enumerate(cells):
            text = c.get_text(" ", strip=True)[:50]
            spans = c.find_all("span")
            span_info = [(sp.get("class", []), sp.get_text(strip=True)[:30]) for sp in spans]
            print(f"  Cell {ci}: text='{text}', spans={span_info}")
