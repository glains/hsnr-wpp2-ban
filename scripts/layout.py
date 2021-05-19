#!/usr/bin/env python

from pathlib import Path

print('creating project structure')

Path("../images").mkdir(exist_ok=True)
Path('../labels').mkdir(exist_ok=True)
Path('../colors').mkdir(exist_ok=True)
