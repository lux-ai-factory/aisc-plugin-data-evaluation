"""A local HTTP server with canned answers per path, recording every request: it stands in for the AISC
platform and the system under test (127.0.0.1 only)."""
import http.server
import json
import threading

import pytest


class Stub:
    def __init__(self):
        self.routes, self.seen = {}, []
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                stub.seen.append({"method": self.command, "path": self.path, "body": body,
                                  "headers": {k.lower(): v for k, v in self.headers.items()}})
                plan = stub.routes.get(self.path.split("?")[0]) or [(404, {"detail": "no route"})]
                status, payload = plan.pop(0) if len(plan) > 1 else plan[0]
                if callable(payload):
                    payload = payload(stub.seen[-1])
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PUT = _serve

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.host = f"127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def route(self, path, *responses):
        self.routes[path] = list(responses)

    def stop(self):
        self.server.shutdown()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.stop()
