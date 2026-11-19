#!/usr/bin/env python3
"""
Skate3-VK Online — Lightweight UDP Room-Code Relay Server
Zero external dependencies (Python 3 standard library only).

Usage:
    python relay_server.py --port 37464

Players set `mode=relay`, `host=<server IP>`, `port=37464`, `room=SK8A` in
`mods/skate3_online.ini` to join the same Freeskate room without port forwarding.
"""

import argparse
import socket
import struct
import time

MAGIC = 0x4E4F3353  # "S3ON"
VERSION = 1
MSG_HELLO = 1
MSG_WELCOME = 2
MSG_STATE = 3
MSG_EVENT = 4
MSG_PING = 5
MSG_PONG = 6
MSG_LEAVE = 7

HDR_FMT = "<IHBBII"  # 16 bytes
HDR_SIZE = struct.calcsize(HDR_FMT)


def main():
    ap = argparse.ArgumentParser(description="Skate3-VK UDP Room-Code Relay Server")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=37464)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)
    print(f"[skate3-relay] listening on UDP {args.bind}:{args.port}")

    # rooms[room_code] = { addr: {"peer_id": int, "name": str, "last_seen": float} }
    rooms = {}
    addr_to_room = {}
    seq = 1

    while True:
        now = time.monotonic()
        # Prune peers silent for >15 seconds
        for room_code in list(rooms.keys()):
            peers = rooms[room_code]
            for a in [k for k, v in peers.items() if now - v["last_seen"] > 15.0]:
                info = peers.pop(a)
                addr_to_room.pop(a, None)
                print(f"[skate3-relay] room={room_code} peer={info['peer_id']} ({info['name']}) timed out")
            if not peers:
                del rooms[room_code]

        try:
            data, addr = sock.recvfrom(1500)
        except socket.timeout:
            continue
        except OSError:
            continue

        if len(data) < HDR_SIZE:
            continue
        magic, ver, mtype, peer_id, pseq, ts = struct.unpack_from(HDR_FMT, data, 0)
        if magic != MAGIC or ver != VERSION:
            continue

        if mtype == MSG_HELLO and len(data) >= HDR_SIZE + 32:
            room = data[HDR_SIZE:HDR_SIZE + 8].split(b"\x00", 1)[0].decode("utf-8", "replace").strip() or "SK8A"
            name = data[HDR_SIZE + 8:HDR_SIZE + 32].split(b"\x00", 1)[0].decode("utf-8", "replace").strip() or "Skater"
            peers = rooms.setdefault(room, {})
            if addr not in peers:
                used = {v["peer_id"] for v in peers.values()}
                pid = next((i for i in range(1, 255) if i not in used), 0)
                if pid == 0:
                    continue
                peers[addr] = {"peer_id": pid, "name": name, "last_seen": now}
                addr_to_room[addr] = room
                print(f"[skate3-relay] room={room} joined peer={pid} ({name}) from {addr[0]}:{addr[1]} ({len(peers)} online)")
            else:
                peers[addr]["last_seen"] = now
                pid = peers[addr]["peer_id"]

            welcome = struct.pack(HDR_FMT + "BBH", MAGIC, VERSION, MSG_WELCOME, 0, seq, int(now * 1000) & 0xFFFFFFFF, pid, len(peers), 60)
            seq += 1
            sock.sendto(welcome, addr)

            # Exchange PeerInfo (kind=4) so everyone in the room sees each other's names immediately.
            room_bytes = room.encode("utf-8")[:8].ljust(8, b"\x00")
            for other_addr, other_info in list(peers.items()):
                if other_addr == addr:
                    continue
                # Tell newcomer about existing peer
                ob = other_info["name"].encode("utf-8")[:127]
                ep1 = struct.pack(HDR_FMT, MAGIC, VERSION, MSG_EVENT, other_info["peer_id"], seq, int(now * 1000) & 0xFFFFFFFF) + room_bytes + struct.pack("<HH", 4, len(ob)) + ob.ljust(128, b"\x00")
                seq += 1
                sock.sendto(ep1, addr)
                # Tell existing peer about newcomer
                nb = name.encode("utf-8")[:127]
                ep2 = struct.pack(HDR_FMT, MAGIC, VERSION, MSG_EVENT, pid, seq, int(now * 1000) & 0xFFFFFFFF) + room_bytes + struct.pack("<HH", 4, len(nb)) + nb.ljust(128, b"\x00")
                seq += 1
                sock.sendto(ep2, other_addr)

        elif mtype == MSG_PING and len(data) >= HDR_SIZE + 8:
            pong = struct.pack(HDR_FMT, MAGIC, VERSION, MSG_PONG, 0, pseq, ts) + data[HDR_SIZE:]
            sock.sendto(pong, addr)
            room = addr_to_room.get(addr)
            if room and addr in rooms.get(room, {}):
                rooms[room][addr]["last_seen"] = now

        elif mtype in (MSG_STATE, MSG_EVENT):
            room = addr_to_room.get(addr)
            if not room or room not in rooms or addr not in rooms[room]:
                continue
            rooms[room][addr]["last_seen"] = now
            for other_addr in list(rooms[room].keys()):
                if other_addr != addr:
                    sock.sendto(data, other_addr)

        elif mtype == MSG_LEAVE:
            room = addr_to_room.pop(addr, None)
            if room and room in rooms and addr in rooms[room]:
                info = rooms[room].pop(addr)
                print(f"[skate3-relay] room={room} left peer={info['peer_id']} ({info['name']})")
                for other_addr in list(rooms[room].keys()):
                    sock.sendto(data, other_addr)


if __name__ == "__main__":
    main()
