"""Run the real-hum evaluation. Equivalent to `earworm eval`.

    uv run python eval/run_eval.py [--weights-search]
"""

import sys

from earworm.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
