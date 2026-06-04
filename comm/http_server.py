import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import random


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/infer":
            l = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(l)
            cls = random.choice(["person", "chair", "door", "none"])
            if cls == "none":
                xs = []
            else:
                s = 0.6 + random.random() * 0.4
                xs = [{"cls": cls, "score": s, "cx": random.random(), "cy": random.random(), "w": random.random(), "h": random.random()}]
            out = {"detections": xs}
            b = json.dumps(out).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        self.send_response(404)
        self.end_headers()


def serve(host="127.0.0.1", port=8000):
    s = HTTPServer((host, port), Handler)
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        pass
