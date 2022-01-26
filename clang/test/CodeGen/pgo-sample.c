// Test if PGO sample use passes are invoked.
//
// TODO: remove redundant `-fno-legacy-pass-manager` key from RUN-lines once we move to NewPM by default.
//
// Ensure Pass PGOInstrumentationGenPass is invoked.
// RUN: %clang_cc1 -fno-legacy-pass-manager -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -fdebug-pass-manager -emit-llvm -o - 2>&1 | FileCheck %s

// CHECK: SimplifyCFGPass
// CHECK: SampleProfileLoaderPass

int func(int a) { return a; }
