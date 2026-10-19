# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU-only HTTP fixture. Never an NPU validation or vLLM substitute."""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main():
    port, model_name, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    if mode == "exit":
        raise SystemExit(4)
    completions = 0

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write((json.dumps(value) if content_type == "application/json" else value).encode())

        def do_GET(self):
            if self.path == "/health":
                self.reply({}, 503 if mode == "unready" else 200)
            elif self.path == "/v1/models":
                self.reply({"data": [{"id": "another-server" if mode == "wrong_model" else model_name}]})
            elif self.path == "/metrics":
                if mode == "no_counter":
                    self.reply("# no speculative counter\n", content_type="text/plain")
                    return
                value = 9 if mode == "stale_counter" else completions * 8
                self.reply(
                    f'vllm:spec_decode_num_drafts_total{{model_name="{model_name}"}} {value}\n',
                    content_type="text/plain",
                )
            else:
                self.reply({}, 404)

        def do_POST(self):
            nonlocal completions
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/completions" or payload["model"] != model_name:
                self.reply({}, 404)
                return
            if mode == "http_error":
                self.reply({"error": "fixture generation failed"}, 500)
                return
            if mode == "slow_request":
                time.sleep(2)
            completions += 1
            self.reply(
                {
                    "choices": [
                        {"index": index, "text": "test", "token_ids": [1, 2], "finish_reason": "length"}
                        for index, _ in enumerate(payload["prompt"])
                    ],
                    "usage": {"completion_tokens": len(payload["prompt"]) * 2},
                }
            )

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
