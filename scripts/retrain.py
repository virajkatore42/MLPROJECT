"""MLOps Continuous Retraining (CT) Pipeline Trigger.
Automates model retraining when triggered by:
- Scheduled cron / CI job
- Automated data drift alert (PSI threshold breach)
- Ingestion of fresh automotive listings
"""

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import main as run_train


def main():
    print("\n" + "=" * 70)
    print("  AUTOMATED CONTINUOUS RETRAINING (CT) TRIGGER")
    print(f"  Triggered at: {datetime.now().isoformat()}")
    print("=" * 70)

    start_time = time.time()
    try:
        run_train()
        duration = round(time.time() - start_time, 2)
        print(f"\n[CT Success] Continuous retraining completed in {duration} seconds.")
    except Exception as e:
        print(f"\n[CT Failure] Retraining pipeline encountered error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
