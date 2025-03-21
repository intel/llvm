// Don't use normal %{run} as we need to control cache directory removal and
// cannot do that reliably when number of devices is unknown.
//
// REQUIRES: level_zero, ocloc
//
// DEFINE: %{cache_vars} = env SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=1 SYCL_CACHE_DIR=%t/cache_dir
// DEFINE: %{build_cmd} = %{build}
// RUN: %{run-aux} mkdir -p %t/cache_dir
//
// The following block of code should be copy-pasted as-is to verify different
// JIT/AOT options. Don't know how to avoid code duplication.
// ******************************
// Check the logs first.
// RUN: %{run-aux} %{build_cmd} -DVALUE=1 -o %t.out
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-WRITE
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ
//
// Now try to substitute the cached image and verify it is actually taken and
// the code/binary there is executed.
// RUN: %{run-aux} mv %t/cache_dir/*/*/*/*/*.bin %t.value1.bin
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{run-aux} %{build_cmd} -DVALUE=2 -o %t.out
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT2
// RUN: %{run-aux} mv %t.value1.bin %t/cache_dir/*/*/*/*/*.bin
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT1
// ******************************
//
// REDEFINE: %{build_cmd} = %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device acm-g10" %s
// ******************************
// Check the logs first.
// RUN: %{run-aux} %{build_cmd} -DVALUE=1 -o %t.out
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-WRITE
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ

//
// Now try to substitute the cached image and verify it is actually taken and
// the code/binary there is executed.
// RUN: %{run-aux} mv %t/cache_dir/*/*/*/*/*.bin %t.value1.bin
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{run-aux} %{build_cmd} -DVALUE=2 -o %t.out
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT2
// RUN: %{run-aux} mv %t.value1.bin %t/cache_dir/*/*/*/*/*.bin
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT1
// ******************************
//
// REDEFINE: %{build_cmd} = %clangxx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device acm-g10,acm-g11" %s
// ******************************
// Check the logs first.
// RUN: %{run-aux} %{build_cmd} -DVALUE=1 -o %t.out
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-WRITE
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s  --check-prefixes=CHECK-CACHE-READ
//
// Now try to substitute the cached image and verify it is actually taken and
// the code/binary there is executed.
// RUN: %{run-aux} mv %t/cache_dir/*/*/*/*/*.bin %t.value1.bin
// RUN: %{run-aux} rm -rf %t/cache_dir/*
// RUN: %{run-aux} %{build_cmd} -DVALUE=2 -o %t.out
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT2
// RUN: %{run-aux} mv %t.value1.bin %t/cache_dir/*/*/*/*/*.bin
// RUN: %{cache_vars} %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefixes RESULT1
// ******************************

// CHECK-CACHE-WRITE: [Persistent Cache]: device binary has been cached
// CHECK-CACHE-READ: [Persistent Cache]: using cached device binary

// RESULT1: Result (0): 1
// RESULT1: Result (1): 1
// RESULT1: Result (2): 1

// RESULT2: Result (0): 2
// RESULT2: Result (1): 2
// RESULT2: Result (2): 2

#include <sycl/detail/core.hpp>

int main() {
  for (int i = 0; i < 3; ++i) {
    sycl::buffer<int, 1> b{1};
    sycl::queue{}
        .submit([&](sycl::handler &cgh) {
          sycl::accessor acc{b, cgh};
          cgh.single_task([=]() { acc[0] = VALUE; });
        })
        .wait();
    std::cout << "Result (" << i << "): " << sycl::host_accessor{b}[0]
              << std::endl;
  }
  return 0;
}
