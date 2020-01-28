#!/usr/bin/env python
# -*- coding: utf-8 -*-
# raise_version_number.py


from __future__ import print_function, division
import os
import sys

from release_tools import check_for_uncommited_changes, replace_version, get_current_version


current_version = str(get_current_version())

# check for new version name as command line argument
try:
    new_version = sys.argv[1]
except IndexError:
    print(f"ERROR: no new version number supplied. Current version is {current_version}", file=sys.stderr)
    sys.exit(1)

# check if new version name differs
if current_version == new_version:
    print("ERROR: new version is the same as old version.", file=sys.stderr)
    sys.exit(1)
print(current_version, new_version)

print("setting version number to", new_version)

files = ["setup.py", "docs/conf.py", "saenopy/__init__.py"]

# Let's go
for file in files:
    replace_version(file, current_version, new_version)
    os.system(f"git add {file}")
    
# commit changes
os.system("git commit -m \"set version to v%s\"" % new_version)
os.system("git tag \"v%s\"" % new_version)