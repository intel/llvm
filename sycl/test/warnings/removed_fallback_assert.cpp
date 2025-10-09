// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// expected-warning@sycl/detail/defines_elementary.hpp:* {{SYCL_FALLBACK_ASSERT has been removed and no longer has any effect.}}
#include <sycl/sycl.hpp>
