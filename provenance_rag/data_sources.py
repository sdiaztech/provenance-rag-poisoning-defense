"""Data sources: MITRE ATT&CK techniques and the CISA KEV catalog."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests


class DataSource(ABC):
    """A dataset file in the data folder, downloaded from its official website if missing."""

    name: str
    filename: str
    url: str
    display_columns: tuple[str, ...] = ()

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / self.filename
        self._table: pd.DataFrame | None = None

    def download(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        response.raise_for_status()
        self.path.write_bytes(response.content)
        print(f"Downloaded {self.url}")

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self.download()
            self._table = self._read()
        return self._table

    @property
    def version(self) -> str:
        return ""

    @property
    def title(self) -> str:
        return f"{self.name} ({self.version})" if self.version else self.name

    @abstractmethod
    def _read(self) -> pd.DataFrame: ...

    def sample(self, rows: int = 5) -> pd.DataFrame:
        columns = [c for c in self.display_columns if c in self.table.columns] or list(self.table.columns)
        return self.table.head(rows)[columns]


class AttackTechniques(DataSource):
    """MITRE ATT&CK Enterprise techniques."""

    release = "20.0"
    name = "MITRE ATT&CK Enterprise techniques"
    filename = f"enterprise-attack-v{release}-techniques.xlsx"
    url = f"https://attack.mitre.org/docs/attack-excel-files/v{release}/enterprise-attack/{filename}"
    display_columns = ("ID", "name", "tactics", "platforms", "is sub-technique", "last modified")

    @property
    def version(self) -> str:
        return f"v{self.release}"

    def _read(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name="techniques")


class KevCatalog(DataSource):
    """CISA Known Exploited Vulnerabilities catalog."""

    name = "CISA Known Exploited Vulnerabilities catalog"
    filename = "known_exploited_vulnerabilities.json"
    url = f"https://www.cisa.gov/sites/default/files/feeds/{filename}"
    display_columns = ("cveID", "vendorProject", "product", "dateAdded", "dueDate", "knownRansomwareCampaignUse")

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)
        self._catalog_version = ""

    @property
    def version(self) -> str:
        self.table
        return f"catalog version {self._catalog_version}"

    def _read(self) -> pd.DataFrame:
        with open(self.path, encoding="utf-8") as f:
            catalog = json.load(f)
        self._catalog_version = catalog.get("catalogVersion", "unknown")
        return pd.DataFrame(catalog["vulnerabilities"])
