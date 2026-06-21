import os
import time
from src import config

def clear_old_uploads(upload_dir: str, max_age_days: int = 7, protect: set | None = None):
    protect = protect or set()
    cutoff = time.time() - max_age_days * 86400
    for fname in os.listdir(upload_dir):
        if fname in protect:
            continue
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)


HTML_PATH = os.path.join(config.root_dir, "static", "monitor_report.html")
STALE_AFTER_SECONDS = config.monitor_stale * 24 * 60 * 60

def is_report_stale(path=HTML_PATH, max_age=STALE_AFTER_SECONDS):
    if not os.path.exists(path):
        return True  # no report yet, definitely stale
    file_age = time.time() - os.path.getmtime(path)
    return file_age > max_age