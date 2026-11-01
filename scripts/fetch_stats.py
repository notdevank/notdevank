#!/usr/bin/env python3
"""Write assets/stats.json from the public GitHub API.

Used by the daily workflow; also safe to run locally. Set GITHUB_TOKEN to
avoid the anonymous rate limit.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
USER = os.environ.get("PROFILE_USER", "notdevank")
API = "https://api.github.com"


def get(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "profile-assets"})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    user = get(f"{API}/users/{USER}")
    repos = get(f"{API}/users/{USER}/repos?per_page=100&sort=updated")

    languages = Counter(repo["language"] for repo in repos if repo["language"])
    ranked = languages.most_common(3)
    total = sum(count for _, count in ranked) or 1

    data = {
        "user": USER,
        "repos": user["public_repos"],
        "stars": sum(repo["stargazers_count"] for repo in repos),
        "followers": user["followers"],
        "languages": [
            {"name": name, "percent": round(count / total * 100)}
            for name, count in ranked
        ],
        "updated": date.today().isoformat(),
    }

    ASSETS.mkdir(exist_ok=True)
    target = ASSETS / "stats.json"
    target.write_text(json.dumps(data, indent=2) + "\n")
    print(target, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
