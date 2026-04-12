"""Tests for webhook delivery."""

from __future__ import annotations

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

import pytest

from lucent_serverless_controller.webhook import deliver_webhook, _deliver


class _Handler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        _Handler.received.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def webhook_server():
    _Handler.received = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestWebhookDelivery:
    def test_delivers_payload(self, webhook_server):
        payload = {"job_id": "j1", "status": "completed"}
        _deliver(webhook_server, payload)
        assert len(_Handler.received) == 1
        assert _Handler.received[0] == payload

    def test_fire_and_forget(self, webhook_server):
        payload = {"job_id": "j2"}
        deliver_webhook(webhook_server, payload)
        import time
        time.sleep(1)
        assert len(_Handler.received) == 1

    def test_retries_on_failure(self):
        _deliver("http://127.0.0.1:1", {"x": 1})


class _FailThenSucceedHandler(BaseHTTPRequestHandler):
    call_count = 0

    def do_POST(self):
        _FailThenSucceedHandler.call_count += 1
        if _FailThenSucceedHandler.call_count < 3:
            self.send_response(500)
        else:
            self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


class TestWebhookRetry:
    def test_retries_until_success(self):
        _FailThenSucceedHandler.call_count = 0
        server = HTTPServer(("127.0.0.1", 0), _FailThenSucceedHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        _deliver(f"http://127.0.0.1:{port}", {"test": True})
        server.shutdown()

        assert _FailThenSucceedHandler.call_count == 3
