/// Test for any deprecated options
// RUN: %clangxx -fsycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-explicit-simd
// RUN: %clangxx -fno-sycl-explicit-simd %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-explicit-simd
// RUN: %clangxx -fsycl -fsycl-allow-device-dependencies %s -### 2>&1 | FileCheck %s -DOPTION=-fsycl-allow-device-dependencies
// RUN: %clangxx -fsycl -fno-sycl-allow-device-dependencies %s -### 2>&1 | FileCheck %s -DOPTION=-fno-sycl-allow-device-dependencies
// CHECK: option '[[OPTION]]' is deprecated and will be removed in a future release

// RUN: %clangxx -fsycl -fsycl-use-bitcode %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_REPLACE -DOPTION=-fsycl-use-bitcode -DOPTION_REPLACE=-fsycl-device-obj=llvmir
// RUN: %clangxx -fsycl -fno-sycl-use-bitcode %s -### 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK_REPLACE -DOPTION=-fno-sycl-use-bitcode -DOPTION_REPLACE=-fsycl-device-obj=spirv
// RUN: %clangxx -fsycl -fsycl-fp32-prec-sqrt %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_REPLACE -DOPTION=-fsycl-fp32-prec-sqrt -DOPTION_REPLACE=-foffload-fp32-prec-sqrt
// RUN: %clangxx -fsycl -fsycl-dump-device-code=/path/to/spv/ %s -### 2>&1 | FileCheck %s --check-prefix=CHECK_REPLACE -DOPTION=-fsycl-dump-device-code=/path/to/spv/ -DOPTION_REPLACE=-save-offload-code=/path/to/spv/
// CHECK_REPLACE: option '[[OPTION]]' is deprecated and will be removed in a future release, use '[[OPTION_REPLACE]]' instead
