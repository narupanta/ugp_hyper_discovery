#!/usr/bin/env python3
"""
syn_force_control_holes.py: Backward-compatible wrapper forwarding to syn_force_control.py with --geometry holes.
"""
import sys
import os

from dataset.synthetic.force_control.syn_force_control import main

if __name__ == "__main__":
    if "--geometry" not in sys.argv:
        sys.argv.extend(["--geometry", "holes"])
    main()
