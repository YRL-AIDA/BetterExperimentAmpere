import json
import re
import sys

PAT = re.compile(
    r"\[(\d+)/\d+\]\s+(\S+)\s+\[(\w+)\]\s+\|\s+([^|]+?)\s+\|\s+status=(\w+)\s+F1=([\d.]+)")


def main():
    if len(sys.argv) < 2:
        print("usage: python extract_undone.py <log.txt> [out.json] [min_index]")
        return
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "undone.json"
    min_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    items, seen = [], set()
    total, taken = 0, 0
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        total += 1
        idx, prompt, fmt, stem, status, f1 = m.groups()
        idx = int(idx)
        if idx < min_index:
            continue
        undone = (status != "ok") or (float(f1) == 0.0 and status != "ok")
        if status != "ok":
            key = (stem.strip(), prompt, fmt)
            if key in seen:
                continue
            seen.add(key)
            items.append({"stem": stem.strip(), "prompt": prompt, "fmt": fmt})
            taken += 1
    json.dump(items, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print(f"scanned {total} result lines (index >= {min_index})")
    print(f"wrote {taken} undone (status != ok) tasks to {out_path}")
    from collections import Counter
    print("by (prompt,format):",
          dict(Counter((i["prompt"], i["fmt"]) for i in items)))


if __name__ == "__main__":
    main()
