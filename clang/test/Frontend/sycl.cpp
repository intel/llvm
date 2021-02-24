// Test that we disallow -cc1 -fsycl without specifying -fsycl-is-device or
// -fsycl-is-host as well.

// RUN: not %clang_cc1 -fsycl %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -fsycl -fsycl-is-host %s 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: error: invalid argument '-fsycl' only allowed with '-fsycl-is-device' or '-fscyl-is-host'
// ERROR: error: invalid argument '-fsycl' only allowed with '-fsycl-is-device' or '-fscyl-is-host'

// Test that passing -fsycl-is-device or -fsycl-is-host without passing -fsycl
// still enabled the SYCL language option.
// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -E -dM %s 2>&1 | FileCheck --check-prefix=ENABLED %s
// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2020 -E -dM %s 2>&1 | FileCheck --check-prefix=ENABLED %s

// ENABLED: #define SYCL_LANGUAGE_VERSION 202001

// Test that you cannot pass -sycl-std= without specifying host or device mode.
// RUN: not %clang_cc1 -sycl-std=2020 %s 2>&1 | FileCheck --check-prefix=ERROR-STD %s
// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-host %s 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: error: invalid argument '-sycl-std=' only allowed with '-fsycl-is-device' or '-fscyl-is-host'
// ERROR-STD: error: invalid argument '-sycl-std=' only allowed with '-fsycl-is-device' or '-fscyl-is-host'

// Test that you cannot specify -fscyl-is-device and -fsycl-is-host at the same time.
// RUN: not %clang_cc1 -fsycl-is-device -fsycl-is-host %s 2>&1 | FileCheck --check-prefix=ERROR-BOTH %s

// ERROR-BOTH: error: invalid argument '-fsycl-is-device' not allowed with '-fsycl-is-host'
