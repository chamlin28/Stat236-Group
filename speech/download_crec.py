"""Download one Congress of CREC data using the congressional-record package.

Usage: python download_crec.py 118 [--passes 3] [--start YYYY-MM-DD --end YYYY-MM-DD]
                                   [--keep-all]

Runs day by day. After a day is fully downloaded and parsed to JSON, the bulky
files the parser does not need (pdf/, html/, dip.xml, premis.xml) are deleted
to save disk space. Only json/ and mods.xml are kept. Use --keep-all to turn
this off. Finished days are skipped on later passes, so extra passes only
retry days that failed because of intermittent GovInfo errors.
"""
import argparse, datetime, logging, os, shutil, sys, time
from config import RAW_DIR, date_range

DROP_DIRS = ["pdf", "html"]
DROP_FILES = ["dip.xml", "premis.xml"]


def day_dir(day):
    return os.path.join(RAW_DIR, str(day.year), "CREC-" + day.strftime("%Y-%m-%d"))


def rmtree_retry(path, tries=5):
    """rmtree that tolerates NFS 'Directory not empty' races. Never raises."""
    for i in range(tries):
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(1 + i)
    shutil.rmtree(path, ignore_errors=True)
    return not os.path.exists(path)


def cleanup_day(ddir):
    """Delete files the parser does not need, but only if the day is complete."""
    done_flag = os.path.join(ddir, "json", "JSON_PARSING_DONE")
    if not os.path.exists(done_flag):
        return 0
    freed = 0
    for name in DROP_DIRS:
        p = os.path.join(ddir, name)
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    freed += os.path.getsize(os.path.join(root, f))
            if not rmtree_retry(p):
                logging.warning("Could not fully remove %s, will retry later", p)
    for name in DROP_FILES:
        p = os.path.join(ddir, name)
        if os.path.isfile(p):
            try:
                freed += os.path.getsize(p)
                os.remove(p)
            except OSError as e:
                logging.warning("Could not remove %s (%s)", p, e)
    return freed


def report_unfinished(d0, d1):
    """Count days that are done, marked no-session, or neither (unfinished)."""
    done = nosess = 0
    unfinished = []
    day = d0
    while day <= d1:
        ddir = day_dir(day)
        if os.path.exists(os.path.join(ddir, "json", "JSON_PARSING_DONE")):
            done += 1
        elif os.path.exists(os.path.join(ddir, "NO_SESSION")):
            nosess += 1
        else:
            unfinished.append(day.strftime("%Y-%m-%d"))
        day += datetime.timedelta(days=1)
    logging.info("SUMMARY: %d days done, %d no-session days, %d UNFINISHED",
                 done, nosess, len(unfinished))
    if unfinished:
        logging.info("Unfinished days: %s", " ".join(unfinished))


def main():
    from congressionalrecord.govinfo.downloader import Downloader

    ap = argparse.ArgumentParser()
    ap.add_argument("congress", type=int)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()

    start, end = date_range(args.congress)
    start = args.start or start
    end = args.end or end
    d0 = datetime.datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.datetime.strptime(end, "%Y-%m-%d")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(message)s")
    os.makedirs(RAW_DIR, exist_ok=True)

    for p in range(1, args.passes + 1):
        t0 = time.time()
        freed_total = 0
        logging.info("Congress %d, pass %d/%d, %s to %s",
                     args.congress, p, args.passes, start, end)
        day = d0
        while day <= d1:
            ds = day.strftime("%Y-%m-%d")
            try:
                Downloader(ds, end=ds, do_mode="json", outpath=RAW_DIR)
            except Exception as e:
                logging.warning("%s failed (%s: %s), will retry on next pass",
                                ds, type(e).__name__, e)
            if not args.keep_all:
                try:
                    freed_total += cleanup_day(day_dir(day))
                except Exception as e:
                    logging.warning("%s cleanup failed (%s: %s), continuing",
                                    ds, type(e).__name__, e)
            day += datetime.timedelta(days=1)
        logging.info("Pass %d finished in %.1f min, freed %.2f GB of unneeded files",
                     p, (time.time() - t0) / 60, freed_total / 1e9)
    report_unfinished(d0, d1)


if __name__ == "__main__":
    main()
