import argparse
import sys
import threading
import time
import os
import socket

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def _wait_port_open(host: str, port: int, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            if s.connect_ex((host, port)) == 0:
                return True
        except Exception:
            pass
        finally:
            try:
                s.close()
            except Exception:
                pass
        time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ui-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (args.self_test or args.server or args.optimize or args.all or args.ui_only):
        args.all = True
    if args.self_test:
        from aiglass.core.pipeline import Pipeline, PipelineConfig
        cfg = PipelineConfig()
        p = Pipeline(cfg)
        out = p.self_test()
        print(out)
        return
    if args.server:
        from aiglass.comm.http_server import serve
        serve(args.host, args.port)
        return
    if args.optimize:
        os.environ["AIGLASS_OPTIMIZE_ONLY"] = "1"
        from aiglass.comm.server import run_server
        host = args.host
        if host in ("127.0.0.1", "localhost"):
            host = "0.0.0.0"
        run_server(host, args.port)
        return
    if args.all and not args.ui_only:
        from aiglass.comm.server import run_server
        host = args.host
        if host in ("127.0.0.1", "localhost"):
            host = "0.0.0.0"
        t = threading.Thread(target=run_server, args=(host, args.port), daemon=True)
        t.start()
        ok = _wait_port_open("127.0.0.1", args.port, timeout_s=3.0)
        if not ok:
            print(f"[WARN] 服务端口未就绪: 127.0.0.1:{args.port}。可能是端口被占用或依赖缺失。")
            print(f"[WARN] 可尝试: 1) 更换端口 --port 8001  2) 关闭占用 3) 临时仅启动界面 --ui-only")
    from PySide6.QtWidgets import QApplication
    from aiglass.client.main_window import MainWindow
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
