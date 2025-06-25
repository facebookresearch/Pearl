#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

ROOT_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test"
)
if os.path.isdir(ROOT_TEST_DIR) and ROOT_TEST_DIR not in __path__:
    __path__.append(ROOT_TEST_DIR)