#!/usr/bin/env python3
"""
Safe async request generator for local/staging testing.
Do NOT run this against third-party servers without permission.
"""

import asyncio
import aiohttp
import random
import string
import time
from typing import Optional

# CONFIG — change only for servers you own / have permission to test
TARGET_URLS = [
    "http://195.189.99.7:60764",       # <--- your test servers
    "http://135.181.8.222:15004",
    "http://195.189.99.7:60764"
]
DURATION_SECONDS = None                 # None = run indefinitely
CONCURRENCY = 10                        # max concurrent requests
REQUEST_INTERVAL = 17                   # send request every 17 seconds
METHODS = ["GET", "POST"]               # allowed methods used randomly
BACKOFF_BASE = 0.5                      # exponential backoff base (seconds)
MAX_PAYLOAD_SIZE = 200                  # max random payload length

# Helpers
def random_path():
    # generate random path or choose from a list
    paths = ["/", "/api/ping", "/api/item", "/health"]
    return random.choice(paths)

def random_payload():
    # small random JSON-like payload
    return {
        "id": random.randint(1, 1000),
        "name": "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15))),
        "note": "".join(random.choices(string.ascii_letters + " ", k=random.randint(10, MAX_PAYLOAD_SIZE))),
    }

async def worker(name: int, session: aiohttp.ClientSession, q: asyncio.Queue):
    while True:
        job = await q.get()
        if job is None:
            q.task_done()
            break

        method, url = job
        backoff = BACKOFF_BASE
        for attempt in range(4):  # retry a few times with backoff
            try:
                if method == "GET":
                    async with session.get(url, timeout=10) as resp:
                        status = resp.status
                        text = await resp.text()
                else:
                    payload = random_payload()
                    async with session.post(url, json=payload, timeout=10) as resp:
                        status = resp.status
                        text = await resp.text()

                print(f"[{name}] {method} {url} -> {status}")
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"[{name}] {method} {url} failed (attempt {attempt+1}): {e}; backoff {backoff:.2f}s")
                await asyncio.sleep(backoff)
                backoff *= 2
        q.task_done()

async def main():
    q = asyncio.Queue()
    stop_time = time.time() + DURATION_SECONDS if DURATION_SECONDS else None

    # Producer: schedule jobs every REQUEST_INTERVAL seconds
    async def producer():
        while stop_time is None or time.time() < stop_time:
            # build URL and method for each target
            for target_url in TARGET_URLS:
                path = random_path()
                url = target_url.rstrip("/") + path
                method = random.choice(METHODS)
                await q.put((method, url))
            
            # wait for REQUEST_INTERVAL seconds before next batch
            await asyncio.sleep(REQUEST_INTERVAL)

        # signal workers to stop
        for _ in range(CONCURRENCY):
            await q.put(None)

    timeout = aiohttp.ClientTimeout(total=15)
    conn = aiohttp.TCPConnector(limit=0)  # let semaphore control concurrency

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        # start workers
        workers = [asyncio.create_task(worker(i, session, q)) for i in range(CONCURRENCY)]
        prod = asyncio.create_task(producer())
        # wait
        await prod
        await q.join()
        # make sure workers exit
        await asyncio.gather(*workers, return_exceptions=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user — exiting.")
