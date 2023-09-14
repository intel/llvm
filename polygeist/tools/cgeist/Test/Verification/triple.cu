// RUN: cgeist ---target aarch64-unknown-linux-gnu %s %stdinclude -S -o - | FileCheck %s -check-prefix=MLIR
// RUN: cgeist ---target aarch64-unknown-linux-gnu %s %stdinclude -emit-llvm -S -o - | FileCheck %s -check-prefix=LLVM

// FIXME: Failing possibly due to different configuration in CI
// XFAIL: *

// MLIR:  llvm.target_triple = "aarch64-unknown-linux-gnu"
// LLVM:  target triple = "aarch64-unknown-linux-gnu"

int main() { return 0; }
