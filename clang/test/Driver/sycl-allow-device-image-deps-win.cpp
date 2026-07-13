/// Test that -fsycl-allow-device-image-dependencies on Windows passes
/// /INCLUDE:__imp_<sym> to force-load user import libs.

// REQUIRES: system-windows

/// Create a minimal import lib with a known symbol via llvm-dlltool.
// RUN: echo "LIBRARY test.dll" > %t.def
// RUN: echo "EXPORTS" >> %t.def
// RUN: echo "TestFunc" >> %t.def
// RUN: llvm-dlltool -m i386:x86-64 --input-def %t.def --output-lib %t.lib

/// Check that /INCLUDE:__imp_TestFunc is passed to the linker.
// RUN: %clang_cl -fsycl --offload-new-driver /clang:--sysroot=%S/Inputs/SYCL \
// RUN:          -fsycl-allow-device-image-dependencies /O2 -### -- %s %t.lib 2>&1 \
// RUN:  | FileCheck %s
// CHECK: "/INCLUDE:__imp_TestFunc"
