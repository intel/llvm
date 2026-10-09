/// Test the diagnostics for spir64_gen used as a SYCL target, which is
/// deprecated in favor of intel_gpu_<arch>.

/// Old offloading model: deprecation warning.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN -DOPT=-fsycl-targets=
// RUN: %clangxx -### -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=spir64_gen-unknown-unknown \
// RUN:   -Xsycl-target-backend "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN-TRIPLE
// RUN: %clangxx -### -fsycl --no-offload-new-driver \
// RUN:   --offload-targets=spir64_gen -Xsycl-target-backend "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN -DOPT=--offload-targets=
// RUN: %clangxx -### -fsycl --no-offload-new-driver \
// RUN:   -fsycl-targets=intel_gpu_pvc,spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen "-device dg2" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=WARN,WARN-XS -DOPT=-fsycl-targets=
// WARN: warning: option '[[OPT]]spir64_gen' is deprecated and will be removed in a future release, use '[[OPT]]intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-TRIPLE: warning: option '-fsycl-targets=spir64_gen-unknown-unknown' is deprecated and will be removed in a future release, use '-fsycl-targets=intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-XS: warning: option '-Xsycl-target-backend=spir64_gen' is deprecated and will be removed in a future release, use '-Xsycl-target-backend=intel_gpu_<arch>' instead [-Wdeprecated]

/// -Xsycl-target-* with a spir64_gen triple is diagnosed even when the target
/// is spelled intel_gpu_<arch>.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xsycl-target-backend=spir64_gen "-options -extra" \
// RUN:   -Xsycl-target-frontend=spir64_gen -DFOO \
// RUN:   -Xsycl-target-linker=spir64_gen "-foo" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN-X
// WARN-X-DAG: warning: option '-Xsycl-target-backend=spir64_gen' is deprecated and will be removed in a future release, use '-Xsycl-target-backend=intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-X-DAG: warning: option '-Xsycl-target-frontend=spir64_gen' is deprecated and will be removed in a future release, use '-Xsycl-target-frontend=intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-X-DAG: warning: option '-Xsycl-target-linker=spir64_gen' is deprecated and will be removed in a future release, use '-Xsycl-target-linker=intel_gpu_<arch>' instead [-Wdeprecated]

/// The deprecation warning is controlled by -Wdeprecated.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend "-device pvc" -Wno-deprecated %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-DIAG
// RUN: not %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend "-device pvc" -Werror=deprecated %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WERROR
// WERROR: error: option '-fsycl-targets=spir64_gen' is deprecated

/// New offloading model: error.
// RUN: not %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR
// RUN: not %clangxx -### -fsycl --offload-new-driver \
// RUN:   -fsycl-targets=spir64_gen-unknown-unknown \
// RUN:   -Xsycl-target-backend "-device pvc" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR-TRIPLE
// RUN: not %clangxx -### -fsycl --offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xsycl-target-backend=spir64_gen "-options -extra" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR-XS
// ERR: error: '-fsycl-targets=spir64_gen' is not supported with the new offloading model; use '-fsycl-targets=intel_gpu_<arch>' instead
// ERR-TRIPLE: error: '-fsycl-targets=spir64_gen-unknown-unknown' is not supported with the new offloading model; use '-fsycl-targets=intel_gpu_<arch>' instead
// ERR-XS: error: '-Xsycl-target-backend=spir64_gen' is not supported with the new offloading model; use '-Xsycl-target-backend=intel_gpu_<arch>' instead

/// The other triple-scoped tool options are diagnosed the same way.
// RUN: not %clangxx -### -fsycl --offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xspirv-translator=spir64_gen "-foo" -Xdevice-post-link=spir64_gen "-foo" \
// RUN:   -Xspirv-to-ir-wrapper=spir64_gen "-foo" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR-XTOOL
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xspirv-translator=spir64_gen "-foo" -Xdevice-post-link=spir64_gen "-foo" \
// RUN:   -Xspirv-to-ir-wrapper=spir64_gen "-foo" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WARN-XTOOL
// ERR-XTOOL-DAG: error: '-Xspirv-translator=spir64_gen' is not supported with the new offloading model; use '-Xspirv-translator=intel_gpu_<arch>' instead
// ERR-XTOOL-DAG: error: '-Xdevice-post-link=spir64_gen' is not supported with the new offloading model; use '-Xdevice-post-link=intel_gpu_<arch>' instead
// ERR-XTOOL-DAG: error: '-Xspirv-to-ir-wrapper=spir64_gen' is not supported with the new offloading model; use '-Xspirv-to-ir-wrapper=intel_gpu_<arch>' instead
// WARN-XTOOL-DAG: warning: option '-Xspirv-translator=spir64_gen' is deprecated and will be removed in a future release, use '-Xspirv-translator=intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-XTOOL-DAG: warning: option '-Xdevice-post-link=spir64_gen' is deprecated and will be removed in a future release, use '-Xdevice-post-link=intel_gpu_<arch>' instead [-Wdeprecated]
// WARN-XTOOL-DAG: warning: option '-Xspirv-to-ir-wrapper=spir64_gen' is deprecated and will be removed in a future release, use '-Xspirv-to-ir-wrapper=intel_gpu_<arch>' instead [-Wdeprecated]

/// The error cannot be downgraded with -Wno-deprecated.
// RUN: not %clangxx -### -fsycl --offload-new-driver -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend "-device pvc" -Wno-deprecated %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERR

/// intel_gpu_<arch> is not diagnosed in either model.
// RUN: %clangxx -### -fsycl --no-offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xsycl-target-backend=intel_gpu_pvc "-options -extra" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-DIAG
// RUN: %clangxx -### -fsycl --offload-new-driver -fsycl-targets=intel_gpu_pvc \
// RUN:   -Xsycl-target-backend=intel_gpu_pvc "-options -extra" %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-DIAG
// NO-DIAG-NOT: spir64_gen' is
