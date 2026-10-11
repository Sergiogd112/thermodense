"""Stream one UKSSDC regional WACCM-X yearly archive and keep every daily member byte-exact.

Usage: fetch_archive.py --base-url FTP_DIR/ --tasks tasks.txt --index N --dest DIR
(tasks.txt: one "<archive name> <listed bytes>" line per archive, from the FTP listing).

Members are stored (uncompressed) ZIP local records without a central directory, so the
archive is read sequentially once. Each member's provider CRC-32 is checked, then the
NetCDF bytes are written unchanged inside a .xz container, with a manifest recording
name, CRC-32, SHA-256 and sizes. A year directory appears only after the whole archive
was read and its byte count matched the FTP listing.
"""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import zlib

# The provider's per-request FTP directory (it embeds the requester's address), e.g.
# "ftp://anonymous:@ftp.ukssdc.ac.uk/wdc/<requester>/". Pass --base-url or set
# WACCMX_FTP_BASE; it is deliberately not stored in the repository.
BASE_ENV = "WACCMX_FTP_BASE"
LOCAL_SIG = 0x04034B50
MIN_FREE_BYTES = 25e9


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def store(directory, name, data, crc, offset):
    path = directory / (name + ".xz")
    tmp = directory / ("." + name + ".xz.tmp")
    packed = lzma.compress(data, preset=6)
    with open(tmp, "wb") as f:
        f.write(packed)
    os.replace(tmp, path)
    return {"name": name, "offset": offset, "size": len(data), "crc32": "%08x" % crc,
            "sha256": hashlib.sha256(data).hexdigest(), "xz_size": len(packed)}


def fetch(archive, dest, listed_size, base, threads=2, max_members=None):
    stem = archive[:-4]
    final = dest / stem
    if (final / "COMPLETE.json").exists():
        print(json.dumps({"event": "skip", "archive": archive}), flush=True)
        return
    partial = dest / ("." + stem + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    started = time.time()
    proc = subprocess.Popen(["curl", "-sS", "--fail", "--max-time", "10800", base + archive],
                            stdout=subprocess.PIPE)
    stream, offset, trailing, end, futures = proc.stdout, 0, 0, "eof", []
    with ThreadPoolExecutor(threads) as pool:
        while max_members is None or len(futures) < max_members:
            head = read_exact(stream, 30)
            if not head:
                break
            sig = struct.unpack("<I", head[:4])[0] if len(head) >= 4 else None
            if sig != LOCAL_SIG:
                rest = head
                while chunk := stream.read(1 << 20):
                    rest += chunk if len(rest) < 4096 else b""
                    trailing += len(chunk)
                trailing += len(head)
                end = "non-local record 0x%08x at offset %d" % (sig or 0, offset)
                break
            if len(head) < 30:
                raise ValueError("truncated local header at %d" % offset)
            _, _, flag, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", head)
            if flag & 0x08 or method != 0 or csize != usize:
                raise ValueError("unsupported member at %d: flag=%d method=%d" % (offset, flag, method))
            name = read_exact(stream, nlen).decode()
            read_exact(stream, xlen)
            data = read_exact(stream, usize)
            if len(data) != usize:
                raise ValueError("truncated member " + name)
            if zlib.crc32(data) != crc:
                raise ValueError("CRC mismatch in " + name)
            if shutil.disk_usage(dest).free < MIN_FREE_BYTES:
                raise RuntimeError("stopping: less than 25 GB free on " + str(dest))
            futures.append(pool.submit(store, partial, name, data, crc, offset))
            offset += 30 + nlen + xlen + usize
            pending = [f for f in futures if not f.done()]
            if len(pending) > 2 * threads:
                pending[0].result()
            if len(futures) % 60 == 0:
                print(json.dumps({"event": "progress", "archive": archive, "members": len(futures),
                                  "bytes": offset, "seconds": round(time.time() - started)}), flush=True)
        members = [f.result() for f in futures]
    if max_members is not None:
        proc.kill()
        proc.wait()
        print(json.dumps({"event": "test-stop", "members": members}), flush=True)
        return
    if proc.wait() != 0:
        raise RuntimeError("curl failed for " + archive)
    read = offset + trailing
    if read != listed_size:
        raise ValueError("read %d bytes but listing says %d" % (read, listed_size))
    record = {"archive": archive, "listed_size": listed_size, "bytes_read": read, "members": len(members),
              "first": members[0]["name"] if members else None, "last": members[-1]["name"] if members else None,
              "trailing_bytes": trailing, "stream_end": end, "stored_bytes": sum(m["xz_size"] for m in members),
              "seconds": round(time.time() - started), "source_archive": archive,
              "storage": "each member's original NetCDF bytes, unchanged, in an .xz container (preset 6)"}
    (partial / "manifest.json").write_text(json.dumps(members, indent=1) + "\n")
    (partial / "COMPLETE.json").write_text(json.dumps(record, indent=1) + "\n")
    os.rename(partial, final)
    print(json.dumps({"event": "complete", **{k: v for k, v in record.items() if k != "source"}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True, help="lines of '<archive> <bytes>'")
    p.add_argument("--index", type=int, required=True, help="1-based line in --tasks")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-members", type=int, help="test only: stop after N members, nothing published")
    p.add_argument("--base-url", default=os.environ.get(BASE_ENV),
                   help=f"provider FTP directory ending in '/' (default: ${BASE_ENV})")
    a = p.parse_args()
    if not a.base_url or not a.base_url.endswith("/"):
        p.error(f"--base-url (or ${BASE_ENV}) must give the provider FTP directory, ending in '/'")
    archive, size = a.tasks.read_text().split("\n")[a.index - 1].split()
    fetch(archive, a.dest, int(size), a.base_url, a.threads, a.max_members)


if __name__ == "__main__":
    main()
