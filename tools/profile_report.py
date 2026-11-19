"""Summarise a PS3RECOMP_PROFILE file using the linker map of ps3game.exe.

usage: python tools/profile_report.py <profile.txt> <ps3game.map> [top]
Prints, per thread, CPU time and the hottest functions (recompiled guest
functions appear as fn_XXXXXXXX = their PS3 address).
"""
import bisect
import re
import sys


def load_map(path):
    base = 0x140000000
    syms = []
    sym_re = re.compile(r'^\s*[0-9a-fA-F]{4}:[0-9a-fA-F]{8}\s+(\S+)\s+([0-9a-fA-F]{16})')
    with open(path, errors='replace') as f:
        for line in f:
            m = re.search(r'Preferred load address is ([0-9a-fA-F]+)', line)
            if m:
                base = int(m.group(1), 16)
            m = sym_re.match(line)
            if m:
                syms.append((int(m.group(2), 16) - base, m.group(1)))
    syms.sort()
    return [s[0] for s in syms], [s[1] for s in syms]


def pretty(name):
    m = re.match(r'\?(fn_[0-9A-F]{8})@@', name)
    if m:
        return m.group(1)
    m = re.match(r'\?([^@]+)@([^@]*)@', name)
    if m:
        return f'{m.group(2)}::{m.group(1)}' if m.group(2) else m.group(1)
    return name


def main():
    prof, mapfile = sys.argv[1], sys.argv[2]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    addrs, names = load_map(mapfile)
    threads = []
    calls = []
    cur = None
    with open(prof) as f:
        for line in f:
            if line.startswith('call '):
                m = re.match(r'call (\d+) caller ([0-9A-F]+) count (\d+) micros (\d+) thread (.*)', line)
                calls.append((int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5).strip()))
                cur = None
                continue
            if line.startswith('thread '):
                m = re.match(r'thread (.*) cpu_ms (\d+) samples (\d+)', line)
                cur = {'name': m.group(1), 'cpu': int(m.group(2)), 'samples': int(m.group(3)), 'hits': {}}
                threads.append(cur)
            elif cur is not None and line.strip().startswith('module '):
                _, mod, n = line.split(None, 2)
                cur['hits']['[' + mod + ']'] = cur['hits'].get('[' + mod + ']', 0) + int(n)
            elif cur is not None and line.strip().startswith('guest '):
                _, pc, n = line.split()
                cur.setdefault('guest', []).append((pc, int(n)))
            elif cur is not None and line.strip():
                off, n = line.split()
                off = int(off, 16)
                i = bisect.bisect_right(addrs, off) - 1
                sym = pretty(names[i]) if i >= 0 else '?'
                cur['hits'][sym] = cur['hits'].get(sym, 0) + int(n)
    threads.sort(key=lambda t: -t['cpu'])
    for t in threads:
        if t['samples'] == 0:
            continue
        print(f"{t['name']}: cpu {t['cpu'] / 1000:.1f}s, {t['samples']} samples")
        total = sum(t['hits'].values()) or 1
        for sym, n in sorted(t['hits'].items(), key=lambda kv: -kv[1])[:top]:
            print(f'   {100 * n / total:5.1f}%  {sym}')
        if t.get('guest'):
            gtotal = sum(n for _, n in t['guest']) or 1
            print('   guest pcs: ' + ', '.join(f'{pc} {100 * n / gtotal:.1f}%' for pc, n in t['guest'][:top]))
    if calls:
        print('blocking syscalls (by total time):')
        for n, caller, count, micros, thread in calls[:40]:
            print(f'   {thread[:28]:28s} syscall {n:4d}  caller {caller}  {count:8d} calls  {micros / 1e6:8.2f}s  avg {micros / max(count, 1):9.1f}us')


if __name__ == '__main__':
    main()
