# Overview
This directory contains ESIMD tests which are run on Intel GPU device only.
Some of them can run on host device too, but in general it is not always
possible as some of ESIMD APIs (e.g. memory access via accessors) is not
implemented for the host device.

Tests within this directory has additional preprocessor definitions available.

`ESIMD_TESTS_FULL_COVERAGE` (default: not defined)\
Enable extended coverage and testing logic with significantly increased
compilation and execution time. Extensive tests could be run manually if needed,
but do not affect CI performance for the basic functionality.
