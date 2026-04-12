"""Trivial echo app — no GPU, no models. Proves the platform works end-to-end.

Run:   python app.py
CLI:   lucent-deploy app.py
"""

import asyncio

import lucent_serverless as ls


class EchoApp(ls.App):
    app_id = "echo"
    compute_type = "CPU"
    cpu_flavor = "cpu3c"
    container_disk_gb = 10
    cloud_type = "COMMUNITY"
    keep_alive = 60
    max_concurrency = 16

    def setup(self) -> None:
        print("EchoApp.setup() ran")

    @ls.realtime("/realtime")
    async def echo(self, ws):
        async for message in ws.iter_text():
            await ws.send_text(f"echo: {message}")


if __name__ == "__main__":
    info = EchoApp.spawn()
    print(f"pod ready: {info.pod_url}")

    async def main():
        async with ls.connect(info.app_id, "/realtime") as ws:
            await ws.send("hello from spawn!")
            print(await ws.recv())

    asyncio.run(main())
