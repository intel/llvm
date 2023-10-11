//==- kernel_compiler_constraints.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx  -fsyntax-only -fsycl -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// kernel_bundles with the  new bundle_state::ext_oneapi_source should NOT
// support several member functions. This test checks that

#include <sycl/sycl.hpp>

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER

  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;

  sycl::queue q;
  sycl::context ctx = q.get_context();
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, "");

  // expected-error@+1 {{no matching member function for call to 'contains_specialization_constants'}}
  kbSrc.contains_specialization_constants();

#endif
}