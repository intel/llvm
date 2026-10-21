// REQUIRES: build-mode
// RUN: echo %{full_atomic_flags} | FileCheck %s --check-prefix=COMMON --implicit-check-not=-Wno-unused-command-line-argument %if target-spir %{ --check-prefix=SPIR %} %else %{ --implicit-check-not=-Xspirv-translator --implicit-check-not=-spirv-ext %}
// RUN: FileCheck %s --check-prefix=TARGETS --input-file=%S/load_local.cpp
// RUN: FileCheck %s --check-prefix=TARGETS --input-file=%S/sub_generic_local.cpp
// RUN: FileCheck %s --check-prefix=TARGETS --input-file=%S/exchange.cpp
// RUN: FileCheck %s --check-prefix=TARGETS --input-file=%S/and_generic_local.cpp

// Check the E2E substitution without requiring a device or compiling kernels.
// Non-SPIR targets must not receive unused SPIR-V translator options.
// COMMON: -DFULL_ATOMIC16_COVERAGE -DFULL_ATOMIC32_COVERAGE
// COMMON-SAME: -DFULL_ATOMIC64_COVERAGE -fsycl-device-code-split=per_kernel
// SPIR-SAME: -Xspirv-translator=spir64
// SPIR-SAME: -spirv-ext=+SPV_KHR_bfloat16,+SPV_INTEL_16bit_atomics

// Exclude individual targets, not entire mixed-target build-only jobs.
// TARGETS: // UNSUPPORTED: target-nvidia, target-amd, spirv-backend
