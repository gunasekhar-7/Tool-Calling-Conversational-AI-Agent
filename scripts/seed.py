from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import get_settings  # noqa: E402
from agent.db.seed import reset_and_seed, seed_if_empty  # noqa: E402
from agent.db.session import create_engine, create_session_factory  # noqa: E402


async def main(reset: bool) -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    async with factory() as db:
        if reset:
            await reset_and_seed(db)
            print("seeded demo data (reset)")
        elif await seed_if_empty(db):
            print("seeded demo data")
        else:
            print("seed data already present; use --reset to rebuild")
        await db.commit()
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    asyncio.run(main(parser.parse_args().reset))
