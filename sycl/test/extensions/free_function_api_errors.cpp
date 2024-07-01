//==---- free_function_api_errors.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx  -fsyntax-only -fsycl -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/sycl.hpp>

using namespace sycl;

// A plain function that is not a free function.
void ff_4(int *ptr, int start) {}

bool test_bundle_api_errors(queue Queue) {
  bool Pass = true;
  context Context{Queue.get_context()};
  device Device{Queue.get_device()};
  std::vector<device> Devices{Context.get_devices()};

  // expected-error@+1 {{no matching function for call to 'has_kernel_bundle'}}
  Pass &= has_kernel_bundle<ff_4, bundle_state::executable>(Context);

  // expected-error@+1 {{no matching function for call to 'has_kernel_bundle'}}
  Pass &= has_kernel_bundle<ff_4, bundle_state::executable>(Context, Devices);

  // expected-error@+2 {{no matching function for call to 'get_kernel_bundle'}}
  kernel_bundle Bundle1 =
      get_kernel_bundle<ff_4, bundle_state::executable>(Context);

  // expected-error@+2 {{no matching function for call to 'get_kernel_bundle'}}
  kernel_bundle Bundle2 =
      get_kernel_bundle<ff_4, bundle_state::executable>(Context, Devices);

  // expected-error@+1 {{no matching function for call to 'is_compatible'}}
  Pass &= is_compatible<ff_4>(Device);

  // expected-error@+1 {{use of undeclared identifier 'ext_oneapi_has_kernel'}}
  Pass &= ext_oneapi_has_kernel<ff_4>(Device);

  // expected-error@+1 {{use of undeclared identifier 'ext_oneapi_get_kernel'}}
  kernel Kernel = ext_oneapi_get_kernel<ff_4>();

  return 0;
}
