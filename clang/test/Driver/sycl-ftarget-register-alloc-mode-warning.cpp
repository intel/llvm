// Test warning SYCL -ftarget-register-alloc-mode for pvc

// RUN: %clang -fsycl -Xs "-device pvc -ze-opt-large-register-file" -### %s 2>&1 | FileCheck -check-prefix=LARGE %s
// RUN: %clang -fsycl -Xs "-device pvc -ze-intel-128-GRF-per-thread" -### %s 2>&1 | FileCheck -check-prefix=SMALL %s
// RUN: %clang -fsycl -Xs "-ze-intel-enable-auto-large-GRF-mode -device pvc" -### %s 2>&1 | FileCheck -check-prefix=AUTO %s

// RUN: %clang -fsycl -fsycl-targets=intel_gpu_pvc -Xs "-ze-opt-large-register-file" -### %s 2>&1 | FileCheck -check-prefix=LARGE %s

// RUN: %clang -fsycl -fsycl-targets=intel_gpu_pvc -Xs"-ze-opt-large-register-file" -### %s 2>&1 | FileCheck -check-prefix=LARGE %s

// RUN: %clang -fsycl -Xsycl-target-backend=spir64 "-device pvc -ze-intel-128-GRF-per-thread" -### %s 2>&1 | FileCheck -check-prefix=SMALL %s

// RUN: %clang -fsycl -fsycl-targets=intel_gpu_pvc -Xsycl-target-backend "-ze-opt-large-register-file" -### %s 2>&1 | FileCheck -check-prefix=LARGE %s

// RUN: %clang -fsycl -fsycl-targets=spir64_gen -Xs "-ze-opt-large-register-file -device pvc" -### %s 2>&1 | FileCheck -check-prefix=LARGE %s

// RUN: %clang -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64 "-ze-opt-large-register-file -device pvc" \
// RUN: -Xsycl-target-backend=spir64_gen "-ze-intel-enable-auto-large-GRF-mode -device pvc" -### %s 2>&1 | FileCheck -check-prefixes=LARGE,AUTO %s

// LARGE: warning: using '-ze-opt-large-register-file' to set GRF mode on PVC hardware is deprecated; use '-ftarget-register-alloc-mode=pvc:large'
// SMALL: warning: using '-ze-intel-128-GRF-per-thread' to set GRF mode on PVC hardware is deprecated; use '-ftarget-register-alloc-mode=pvc:small'
// AUTO:  warning: using '-ze-intel-enable-auto-large-GRF-mode' to set GRF mode on PVC hardware is deprecated; use '-ftarget-register-alloc-mode=pvc:auto'
