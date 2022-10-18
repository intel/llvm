// Test that we disallow -fsycl-enable-int-header-diags without -fsycl-is-host.

// RUN: not %clang_cc1 -fsycl-enable-int-header-diags %s 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not %clang_cc1 -fsycl-is-device -fsycl-enable-int-header-diags %s 2>&1 | FileCheck --check-prefix=ERROR %s

// ERROR: error: option '-fsycl-enable-int-header-diags' cannot be specified without '-fsycl-is-host'
