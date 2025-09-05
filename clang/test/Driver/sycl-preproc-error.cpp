/// Allow for preprocessing to 'succeed' even though there are coding issues
/// in the source.  We always have an additional step to generate the
/// integration header and footer, so if that fails we still want to produce
/// preprocessing information in the subsequent passes.
// RUN: %clang -fsycl -fno-sycl-use-footer -E -dM %s 2>&1 | FileCheck %s --check-prefix=PP_CHECK
// PP_CHECK: SYCL_PP_CHECK

void foo(;
#define SYCL_PP_CHECK
