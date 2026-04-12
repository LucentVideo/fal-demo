"""Trivial echo app — no GPU, no models. Proves the platform works end-to-end.

Deploy:  python app.py
Use:     async with ls.connect("echo") as ws: ...
"""

import lucent_serverless as ls


class EchoApp(ls.App):
    app_id = "echo"
    image_ref = "raylightdimi/lucent-echo:latest"
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
    from dotenv import load_dotenv

    load_dotenv()
    ls.deploy(EchoApp)
