"""Trivial echo app — no GPU, no models. Proves the runner mounts ws routes."""

import lucent_serverless as ls


class EchoApp(ls.App):
    image_ref = "lucent-serverless/echo:dev"
    machine_type = "cpu"
    keep_alive = 60
    max_concurrency = 16

    def setup(self) -> None:
        print("EchoApp.setup() ran")

    @ls.realtime("/realtime")
    async def echo(self, ws):
        async for message in ws.iter_text():
            await ws.send_text(f"echo: {message}")
