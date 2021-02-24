// Test that we disallow -cc1 -fsycl, even when specifying device or host mode.

// RUN: not %clang_cc1 -fsycl %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang_cc1 -fsycl -fsycl-is-device %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang_cc1 -fsycl -fsycl-is-host %s 2>&1 | FileCheck --check-prefix=ERROR %s

// ERROR: error: unknown argument: '-fsycl'

// Test that you cannot pass -sycl-std= without specifying host or device mode.
// RUN: not %clang_cc1 -sycl-std=2020 %s 2>&1 | FileCheck --check-prefix=ERROR-STD %s
// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-device %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -sycl-std=2020 -fsycl-is-host %s 2>&1 | FileCheck --allow-empty %s

// CHECK-NOT: error: invalid argument '-sycl-std=' only allowed with '-fsycl-is-device' or '-fscyl-is-host'
// ERROR-STD: error: invalid argument '-sycl-std=' only allowed with '-fsycl-is-device' or '-fscyl-is-host'

// Test that you cannot specify -fscyl-is-device and -fsycl-is-host at the same time.
// RUN: not %clang_cc1 -fsycl-is-device -fsycl-is-host %s 2>&1 | FileCheck --check-prefix=ERROR-BOTH %s

// ERROR-BOTH: error: invalid argument '-fsycl-is-device' not allowed with '-fsycl-is-host'
