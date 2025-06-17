// RUN: %clangxx -DSYCL_FALLBACK_ASSERT=1 %fsycl-host-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// expected-warning@sycl/detail/defines_elementary.hpp:* {{SYCL_FALLBACK_ASSERT is deprecated.}}
#include <sycl/sycl.hpp>
