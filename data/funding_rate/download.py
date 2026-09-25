"""Download all BTC funding-rate monthly files from data.binance.vision and merge per symbol."""
import csv, io, re, urllib.request, zipfile, os, hashlib

BASE = "https://data.binance.vision/"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix={}&delimiter=/"
TARGETS = {
    "BTCUSDT": "data/futures/um/monthly/fundingRate/BTCUSDT/",
    "BTCUSDC": "data/futures/um/monthly/fundingRate/BTCUSDC/",
    "BTCUSD_PERP": "data/futures/cm/monthly/fundingRate/BTCUSD_PERP/",
}
OUT = os.path.dirname(os.path.abspath(__file__))

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()

for sym, prefix in TARGETS.items():
    keys = sorted(re.findall(r"<Key>([^<]*\.zip)</Key>", get(LIST.format(prefix)).decode()))
    header, rows = None, []
    for k in keys:
        blob = get(BASE + k)
        expected = get(BASE + k + ".CHECKSUM").decode().split()[0]
        assert hashlib.sha256(blob).hexdigest() == expected, k
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            text = z.read(z.namelist()[0]).decode()
        r = list(csv.reader(io.StringIO(text)))
        if r and not r[0][0].isdigit():
            header = header or r[0]
            r = r[1:]
        rows += r
    rows = sorted({tuple(x) for x in rows if x}, key=lambda x: int(x[0]))
    path = os.path.join(OUT, sym + "_fundingRate.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(sym, len(keys), "files ->", len(rows), "rows", path)
