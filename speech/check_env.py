"""Pre-flight check. Run once in the new environment, ideally inside the
interactive session: python check_env.py
"""
import importlib, os, sys

print("Python", sys.version.split()[0])
ok = True
for mod in ["pandas", "pyarrow", "congressionalrecord", "bs4", "lxml", "urllib3", "certifi"]:
    try:
        importlib.import_module(mod)
        print("  OK     ", mod)
    except Exception as e:
        ok = False
        print("  MISSING", mod, "->", type(e).__name__, e)

try:
    from congressionalrecord.govinfo.downloader import Downloader  # noqa
    print("  OK      Downloader import")
except Exception as e:
    ok = False
    print("  FAILED  Downloader import ->", type(e).__name__, e)

from config import DATA_DIR, RAW_DIR, OUT_DIR
print("DATA_DIR:", DATA_DIR)
try:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t = os.path.join(DATA_DIR, ".write_test")
    open(t, "w").close(); os.remove(t)
    print("  OK      data directory is writable")
except Exception as e:
    ok = False
    print("  FAILED  data directory ->", e)

try:
    import urllib3, certifi
    http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(),
                               timeout=urllib3.Timeout(connect=5, read=15))
    r = http.request("GET", "https://www.govinfo.gov/metadata/pkg/CREC-2025-01-03/mods.xml")
    print("  govinfo.gov status:", r.status, "| final URL:", r.geturl())
except Exception as e:
    ok = False
    print("  FAILED  govinfo.gov ->", type(e).__name__, e)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED, see above")
