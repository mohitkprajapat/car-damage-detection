import time
import os

def clear_old_uploads(upload_dir: str, max_age_days: int = 7):
    cutoff = time.time() - max_age_days * 86400
    for fname in os.listdir(upload_dir):
        fpath = os.path.join(upload_dir, fname)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            os.remove(fpath)