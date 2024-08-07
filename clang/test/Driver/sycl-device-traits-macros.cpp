/// Check that no device traits macros are defined if sycl is disabled: 
// RUN:   %clang -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DISABLED %s
// CHECK-DISABLED-NOT: "{{.*}}SYCL_ANY_DEVICE_HAS{{.*}}"
// CHECK-DISABLED-NOT: "{{.*}}SYCL_ALL_DEVICES_HAVE{{.*}}"

/// Check device traits macros are defined if sycl is enabled: 
/// In this case, where no specific sycl targets are passed, the sycl
/// targets are spir64 and the host target (e.g. x86_64). We expect two
/// occurrences of the macro definition, one for host and one for device.
// RUN:   %clang -fsycl --no-offload-new-driver -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ENABLED %s
// RUN:   %clang -fsycl --offload-new-driver -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ENABLED %s
// CHECK-ENABLED-COUNT-2: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"

/// Check device traits macros are defined if sycl is enabled: 
/// In this case the sycl targets are spir64, spir64_gen and the host
/// target (e.g. x86_64). We expect three occurrences of the macro
/// definition, one for host and one for each of the two devices.
// RUN:   %clang -fsycl --no-offload-new-driver -fsycl-targets=spir64,spir64_gen -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-TARGETS %s
// RUN:   %clang -fsycl --offload-new-driver -fsycl-targets=spir64,spir64_gen -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-TARGETS %s
// CHECK-SYCL-TARGETS-COUNT-3: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"

/// Check device traits macros are defined if sycl is enabled:
/// In this case, no specific sycl targets are passed, and `-fsycl-device-only`
/// is provided for device compilation only with no `fsycl`, the only sycl
/// target is the default spir64 without a host target. Hence, we expect only
/// one occurrence of the macro definition (for the device target).
// RUN:   %clang -fsycl-device-only -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-DEVICE-ONLY %s
// CHECK-SYCL-DEVICE-ONLY-COUNT-1: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"

/// Check device traits macros are defined if sycl is enabled:
/// Verify that when compiling for multiple targets and maySupportOtherAspects
/// is false for all of the targets, the driver will never add the
/// __SYCL_ANY_DEVICE_HAS_ANY_ASPECT__ macro to the compilation arguments.
/// NOTE: Both intel_gpu_pvc and amd_gpu_gfx906 have non-empty aspects lists and
/// set maySupportOtherAspects to false, hence why they are used for this test.
// RUN: %clangxx -fsycl --no-offload-new-driver -nogpulib -fsycl-targets=intel_gpu_pvc,amd_gpu_gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-TARGETS-NO-MAY-SUPPORT-OTHER-ASPECTS %s
// RUN: %clangxx -fsycl --offload-new-driver -nogpulib -fsycl-targets=intel_gpu_pvc,amd_gpu_gfx906 \
// RUN:   -fsycl-libspirv-path=%S/Inputs/SYCL/libspirv.bc -### %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-SYCL-TARGETS-NO-MAY-SUPPORT-OTHER-ASPECTS %s
// CHECK-SYCL-TARGETS-NO-MAY-SUPPORT-OTHER-ASPECTS-NOT: "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1"
// CHECK-SYCL-TARGETS-NO-MAY-SUPPORT-OTHER-ASPECTS: "-D__SYCL_ANY_DEVICE_HAS_[[ASPECTi:[a-z0-9_]+]]__=1"
// CHECK-SYCL-TARGETS-NO-MAY-SUPPORT-OTHER-ASPECTS: "-D__SYCL_ALL_DEVICES_HAVE_[[ASPECTj:[a-z0-9_]+]]__=1"
