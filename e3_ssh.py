"""4090 主机 SSH/SFTP 可编程助手（经本地 HTTP CONNECT 代理 127.0.0.1:18080）。

用法（命令行）：
  python e3_ssh.py run '<shell cmd>'        # 远程执行并打印 stdout/stderr/rc
  python e3_ssh.py put <local> <remote>     # 经 SFTP 上传
  python e3_ssh.py get <remote> <local>     # 经 SFTP 下载

模块级：SSH 已缓存单例，多次调用复用。
"""
from __future__ import annotations
import socket, sys

import paramiko as _paramiko
from paramiko import SSHClient

HOST = "hn01-ssh.gpuhome.cc"
PORT = 30594
USER = "root"
PASS = "1a0vv0c1"
PROXY = ("127.0.0.1", 18080)

_cli = None


def _tunnel_sock():
    """经本地 HTTP CONNECT 代理打通到 4090:PORT 的 TCP socket。"""
    s = socket.create_connection(PROXY, timeout=15)
    req = f"CONNECT {HOST}:{PORT} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n\r\n"
    s.sendall(req.encode())
    s.settimeout(15)
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            raise RuntimeError("proxy closed during CONNECT")
        buf += chunk
    status = buf.split(b"\r\n", 1)[0].decode(errors="ignore")
    if "200" not in status:
        raise RuntimeError(f"CONNECT failed: {status}")
    s.settimeout(None)
    return s


def _client():
    global _cli
    if _cli is not None:
        return _cli
    c = SSHClient()
    c.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              sock=_tunnel_sock(), timeout=30, banner_timeout=30,
              auth_timeout=30)
    _cli = c
    return c


def run(cmd: str, timeout: float = 600) -> tuple[str, str, int]:
    c = _client()
    _, out, err = c.exec_command(cmd, timeout=timeout, get_pty=False)
    o = out.read().decode(errors="replace")
    e = err.read().decode(errors="replace")
    rc = out.channel.recv_exit_status()
    return o, e, rc


def put(local: str, remote: str) -> None:
    c = _client()
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()


def get(remote: str, local: str) -> None:
    c = _client()
    sftp = c.open_sftp()
    sftp.get(remote, local)
    sftp.close()


def _banner():
    o, e, rc = run("echo CONNECTED-$(hostname); nvidia-smi -L 2>/dev/null | head -3; "
                   "ls -d /root/srpc_e2 2>/dev/null; "
                   "python3 -c 'import torch,numpy;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())' 2>&1")
    print(o)
    print("STDERR:", e[:500])
    print("rc:", rc)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        o, e, rc = run(" ".join(sys.argv[2:]))
        print(o)
        if e:
            print("STDERR:", e)
        print("rc:", rc)
    elif len(sys.argv) == 4 and sys.argv[1] == "put":
        put(sys.argv[2], sys.argv[3])
        print("put ok")
    elif len(sys.argv) == 4 and sys.argv[1] == "get":
        get(sys.argv[2], sys.argv[3])
        print("get ok")
    else:
        _banner()