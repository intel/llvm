/// Tests that -fsycl-targets=spir64_gen errors in the new offload model and
/// still works in the old offload model. The intel_gpu_* equivalents are clean
/// in both models.

// --- New offload model ---

// spir64_gen should error in the new offload model.
// RUN: not %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64_gen \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=DEPRECATED

// -Xsycl-target-backend=spir64_gen should also error in the new offload model.
// RUN: not %clangxx --offload-new-driver -fsycl -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device pvc" -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEPRECATED

// The new per-device style should be clean in the new offload model.
// RUN: %clangxx --offload-new-driver -fsycl \
// RUN:   -fsycl-targets=intel_gpu_pvc,intel_gpu_dg2 \
// RUN:   -Xsycl-target-backend=intel_gpu_pvc "-options_for_pvc" \
// RUN:   -Xsycl-target-backend=intel_gpu_dg2 "-options_for_dg" \
// RUN:   -### %s 2>&1 | FileCheck %s --check-prefix=NO-ERROR

// --- Old offload model ---

// spir64_gen should still work in the old offload model (no error).
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-ERROR

// -Xsycl-target-backend=spir64_gen should also work in the old offload model.
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device pvc" -### %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-ERROR

// DEPRECATED: SYCL target 'spir64_gen' is no longer supported
// NO-ERROR-NOT: SYCL target 'spir64_gen' is no longer supported
