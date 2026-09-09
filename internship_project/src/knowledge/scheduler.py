"""
scheduler.py — Periodic knowledge base update scheduler for Task 3.
Uses APScheduler for lightweight background scheduling.
Implements the periodic update mechanism requirement.

The scheduler can be configured to:
1. Watch a local directory for new files and auto-ingest them
2. Run on a configurable interval (default: every 30 minutes)
3. Be started/stopped from the Streamlit app

Usage:
    from src.knowledge.scheduler import KnowledgeScheduler
    scheduler = KnowledgeScheduler(watch_dir="data/knowledge_inbox")
    scheduler.start()
    # ... app runs ...
    scheduler.stop()
"""
import os
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class KnowledgeScheduler:
    """
    Background scheduler for periodic knowledge base updates.
    Watches a directory for new files and ingests them automatically.
    """

    def __init__(
        self,
        watch_dir: str = "data/knowledge_inbox",
        interval_minutes: int = 30,
        on_update: Optional[Callable] = None,
    ):
        self.watch_dir = watch_dir
        self.interval_minutes = interval_minutes
        self.on_update = on_update  # Optional callback after successful update
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._processed_files: set = set()
        self._log: list = []

        os.makedirs(watch_dir, exist_ok=True)
        self._load_processed_registry()

    def _load_processed_registry(self):
        registry_path = os.path.join(self.watch_dir, ".processed_registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    self._processed_files = set(json.load(f))
            except Exception:
                self._processed_files = set()

    def _save_processed_registry(self):
        registry_path = os.path.join(self.watch_dir, ".processed_registry.json")
        with open(registry_path, "w") as f:
            json.dump(list(self._processed_files), f)

    def _scan_and_ingest(self):
        """Scan watch directory and ingest any new files."""
        from src.knowledge.updater import update_from_file
        
        new_files = []
        try:
            for fname in os.listdir(self.watch_dir):
                if fname.startswith("."):
                    continue
                fpath = os.path.join(self.watch_dir, fname)
                if os.path.isfile(fpath) and fpath not in self._processed_files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in {".txt", ".md", ".pdf"}:
                        new_files.append(fpath)
        except Exception as e:
            logger.error(f"Error scanning watch directory: {e}")
            return

        for fpath in new_files:
            logger.info(f"[Scheduler] Ingesting new file: {fpath}")
            result = update_from_file(fpath)
            self._processed_files.add(fpath)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "file": os.path.basename(fpath),
                "chunks_added": result.get("chunks_added", 0),
                "success": result.get("success", False),
                "message": result.get("message", ""),
            }
            self._log.append(log_entry)
            if self.on_update and result.get("success"):
                self.on_update(log_entry)

        if new_files:
            self._save_processed_registry()

    def _run_cycle(self):
        """Run one scan cycle and schedule the next."""
        if self._running:
            try:
                self._scan_and_ingest()
            except Exception as e:
                logger.error(f"Scheduler cycle error: {e}")
            finally:
                if self._running:
                    self._timer = threading.Timer(
                        self.interval_minutes * 60, self._run_cycle
                    )
                    self._timer.daemon = True
                    self._timer.start()

    def start(self):
        """Start the background scheduler."""
        if not self._running:
            self._running = True
            self._timer = threading.Timer(
                self.interval_minutes * 60, self._run_cycle
            )
            self._timer.daemon = True
            self._timer.start()
            logger.info(
                f"[Scheduler] Started. Watching '{self.watch_dir}' "
                f"every {self.interval_minutes} minutes."
            )

    def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        logger.info("[Scheduler] Stopped.")

    def run_now(self):
        """Manually trigger an immediate scan (for the UI 'Update Now' button)."""
        self._scan_and_ingest()
        return self.get_log()

    def get_log(self) -> list:
        """Return the update activity log."""
        return list(reversed(self._log))[-20:]  # Last 20 entries

    def get_status(self) -> dict:
        """Return current scheduler status."""
        return {
            "running": self._running,
            "watch_dir": self.watch_dir,
            "interval_minutes": self.interval_minutes,
            "processed_files": len(self._processed_files),
            "log_entries": len(self._log),
        }
