# Copyright (C) 2024 Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import re


@functools.total_ordering
class Version:
    def __init__(self, version: str):
        assert isinstance(version, (str, Version))
        if isinstance(version, str):
            self.str = version
            match = re.match(r"^(\d+)\.(\d+)$", self.str)
            assert isinstance(match, re.Match)
            self.major = int(match.groups()[0])
            self.minor = int(match.groups()[1])
        else:
            self.str = version.str
            self.major = version.major
            self.minor = version.minor

    def __eq__(self, other) -> bool:
        assert isinstance(other, Version)
        return self.major == other.major and self.minor == other.minor

    def __lt__(self, other) -> bool:
        if not isinstance(other, Version):
            import ipdb

            ipdb.set_trace()
        return self.major < other.major or (
            self.major == other.major and self.minor < other.minor
        )

    def __str__(self) -> str:
        return self.str
