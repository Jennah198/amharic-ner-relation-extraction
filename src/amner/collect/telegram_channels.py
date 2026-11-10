"""Telethon scraper for public Amharic news Telegram channels.

Not run by default because it needs personal API credentials. To use it:

1. Create an app at https://my.telegram.org -> "API development tools".
2. Put the credentials in a ``.env`` file at the project root (never commit it)::

       TG_API_ID=123456
       TG_API_HASH=0123456789abcdef0123456789abcdef

3. Run (the first run asks for your phone number + login code)::

       python -m amner.collect.telegram_channels --channels tikvahethiopia EBCworld --limit 2000

Output: ``data/raw/telegram.jsonl`` with one message per line. The output has
the same schema as the BBC scraper, so ``amner.build_corpus`` picks it up
automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv


async def scrape(channels: list[str], limit: int, out: Path) -> int:
    from telethon import TelegramClient  # imported lazily: optional dependency

    load_dotenv()
    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    async with TelegramClient("amner_session", api_id, api_hash) as client:
        with out.open("a", encoding="utf-8") as f:
            for ch in channels:
                async for msg in client.iter_messages(ch, limit=limit):
                    if not msg.message:
                        continue
                    f.write(json.dumps({
                        "source": f"telegram:{ch}",
                        "url": f"https://t.me/{ch}/{msg.id}",
                        "title": "",
                        "date": msg.date.isoformat() if msg.date else "",
                        "text": msg.message,
                    }, ensure_ascii=False) + "\n")
                    n += 1
                print(f"[telegram] {ch}: total {n}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/raw/telegram.jsonl"))
    a = ap.parse_args()
    print(asyncio.run(scrape(a.channels, a.limit, a.out)), "messages saved")


if __name__ == "__main__":
    main()
