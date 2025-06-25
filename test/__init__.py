#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pkgutil import extend_path
from pathlib import Path

ROOT_TEST_DIR = Path(__file__).resolve().parents[2] / "test"
__path__ = list(extend_path(__path__, __name__))
if str(ROOT_TEST_DIR) not in __path__:
    __path__.append(str(ROOT_TEST_DIR))