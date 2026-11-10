"""Polite BeautifulSoup scraper for BBC News Amharic (https://www.bbc.com/amharic).

Strategy
--------
1. Start from the Amharic homepage and the politics/Ethiopia topic pages and
   walk their ``?page=N`` pagination to discover article URLs
   (``/amharic/articles/<id>``).
2. Fetch each article, extract headline, publication date and body paragraphs.
3. Write one JSON object per article to ``data/raw/bbc_amharic.jsonl``.

Etiquette: robots.txt is checked with ``urllib.robotparser``, a descriptive
User-Agent is sent, requests are rate-limited (``--delay``) and already scraped
URLs are skipped so the scraper can be resumed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

BASE = "https://www.bbc.com"
UA = "AmharicNER-StudentProject/1.0 (academic research; contact via GitHub)"
ARTICLE_RE = re.compile(r"/amharic/articles/[a-z0-9]+")
TOPIC_RE = re.compile(r"/amharic/topics/[a-z0-9]+")

# Seed listing pages: homepage + topic pages discovered on it.
SEEDS = ["/amharic"]


class BBCAmharicScraper:
    def __init__(self, out_path: Path, delay: float = 1.0, timeout: int = 20):
        self.out_path = out_path
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA
        self.rp = robotparser.RobotFileParser(BASE + "/robots.txt")
        self.rp.read()
        self.seen: set[str] = set()
        if out_path.exists():
            for line in out_path.open(encoding="utf-8"):
                self.seen.add(json.loads(line)["url"])

    def get(self, path: str) -> str | None:
        url = path if path.startswith("http") else BASE + path
        if not self.rp.can_fetch(UA, url):
            print(f"[robots] skip {url}")
            return None
        time.sleep(self.delay)
        try:
            r = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as e:
            print(f"[error] {url}: {e}")
            return None
        if r.status_code != 200:
            print(f"[http {r.status_code}] {url}")
            return None
        return r.text

    def discover(self, max_pages_per_topic: int, max_topics: int) -> list[str]:
        articles: list[str] = []
        topics: list[str] = []
        for seed in SEEDS:
            html = self.get(seed) or ""
            articles += ARTICLE_RE.findall(html)
            topics += TOPIC_RE.findall(html)
        topics = list(dict.fromkeys(topics))[:max_topics]
        for t in topics:
            for page in range(1, max_pages_per_topic + 1):
                html = self.get(f"{t}?page={page}")
                if not html:
                    break
                found = ARTICLE_RE.findall(html)
                if not found:
                    break
                articles += found
            print(f"[discover] {t}: total so far {len(set(articles))}")
        return list(dict.fromkeys(articles))

    @staticmethod
    def parse_article(html: str, url: str) -> dict | None:
        soup = BeautifulSoup(html, "lxml")
        main = soup.find("main") or soup
        h1 = main.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else ""
        paras = []
        for p in main.find_all("p"):
            # skip figure captions / bylines / promos
            if p.find_parent(["figure", "figcaption", "aside", "nav", "footer"]):
                continue
            t = p.get_text(" ", strip=True)
            if t:
                paras.append(t)
        date = ""
        tm = soup.find("time")
        if tm and tm.get("datetime"):
            date = tm["datetime"]
        else:
            m = re.search(r'"datePublished":"([^"]+)"', html)
            date = m.group(1) if m else ""
        if not paras:
            return None
        return {"source": "bbc_amharic", "url": url, "title": title,
                "date": date, "text": "\n".join(paras)}

    def run(self, max_articles: int, max_pages_per_topic: int, max_topics: int) -> int:
        urls = [u for u in self.discover(max_pages_per_topic, max_topics) if BASE + u not in self.seen]
        print(f"[run] {len(urls)} new article URLs, fetching up to {max_articles}")
        n = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            for path in urls[:max_articles]:
                html = self.get(path)
                if not html:
                    continue
                doc = self.parse_article(html, BASE + path)
                if doc:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 25 == 0:
                        print(f"[run] saved {n}")
        return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw/bbc_amharic.jsonl"))
    ap.add_argument("--max-articles", type=int, default=600)
    ap.add_argument("--max-pages-per-topic", type=int, default=10)
    ap.add_argument("--max-topics", type=int, default=12)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    n = BBCAmharicScraper(a.out, delay=a.delay).run(a.max_articles, a.max_pages_per_topic, a.max_topics)
    print(f"done: {n} articles -> {a.out}")


if __name__ == "__main__":
    main()
