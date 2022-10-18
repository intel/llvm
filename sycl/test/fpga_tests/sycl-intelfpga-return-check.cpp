// REQUIRES: system-linux

/// Tests behavior with aoc -fintelpga.
/// Uses a dummy aoc which returns '42' to make sure we properly emit a
/// diagnostic and also do not stop compilation
// RUN: env PATH=%S/Inputs:$PATH \
// RUN: not %clangxx -fsycl -fintelfpga -Xshardware %s -v 2>&1 \
// RUN:  | FileCheck %s -check-prefix ERROR_OUTPUT
// ERROR_OUTPUT: ld{{.*}} -o a.out
// ERROR_OUTPUT: The FPGA image generated during this compile contains timing
// ERROR_OUTPUT-SAME: violations

int main() { return 0; }
