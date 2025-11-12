// REQUIRES: ocloc, gpu, target-spir
// Test with `--offload-new-driver`
//
// RUN: %clangxx -fsycl -fsycl-device-code-split=per_source \
// RUN:   -fsycl-targets=spir64_gen \
// RUN:   -Xsycl-target-backend=spir64_gen \
// RUN:   "-device tgllp" -I %S/Inputs -o %t.out \
// RUN:   %S/split-per-source-main.cpp \
// RUN:   %S/Inputs/split-per-source-second-file.cpp \
// RUN:   -fsycl-dead-args-optimization --offload-new-driver
// RUN: %{run} %t.out
//
// Check that target is passed to sycl-post-link for filtering
// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_pvc,intel_gpu_dg1,spir64_x86_64 \
// RUN: --offload-new-driver \
// RUN: -v %S/Inputs/aot.cpp 2>&1 | FileCheck %s --check-prefix=CHECK_TOOLS_FILTER
//
// CHECK_TOOLS_FILTER-DAG: sycl-post-link{{.*}} -o intel_gpu_pvc,{{.*}}
// CHECK_TOOLS_FILTER-DAG: sycl-post-link{{.*}} -o intel_gpu_dg1,{{.*}}
// CHECK_TOOLS_FILTER-DAG: sycl-post-link{{.*}} -o spir64_x86_64,{{.*}}
