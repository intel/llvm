//==- kernel_compiler_constraints.cpp --------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx  -fsyntax-only -fsycl -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

// kernel_bundles sporting the new bundle_state::ext_oneapi_source should NOT
// support several member functions. This test confirms that.

#include <sycl/sycl.hpp>

int main() {
#ifdef SYCL_EXT_ONEAPI_KERNEL_COMPILER

  namespace syclex = sycl::ext::oneapi::experimental;
  using source_kb = sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>;

  sycl::queue q;
  sycl::context ctx = q.get_context();
  std::vector<sycl::device> devices = ctx.get_devices();
  source_kb kbSrc = syclex::create_kernel_bundle_from_source(
      ctx, syclex::source_language::opencl, "");

  // expected-error@+1 {{no matching member function for call to 'contains_specialization_constants'}}
  kbSrc.contains_specialization_constants();

  // expected-error@+1 {{no matching member function for call to 'native_specialization_constant'}}
  kbSrc.native_specialization_constant();

  constexpr sycl::specialization_id<int> SpecName;
  // expected-error@+1 {{no matching member function for call to 'has_specialization_constant'}}
  kbSrc.has_specialization_constant<SpecName>();

  // expected-error@+1 {{no matching member function for call to 'get_specialization_constant'}}
  auto i = kbSrc.get_specialization_constant<SpecName>();

  // expected-error@+1 {{no matching member function for call to 'get_kernel'}}
  kbSrc.get_kernel<class kid>();

  // expected-error@+1 {{no matching member function for call to 'get_kernel_ids'}}
  std::vector<sycl::kernel_id> vec = kbSrc.get_kernel_ids();

  class TestKernel1;
  sycl::kernel_id TestKernel1ID = sycl::get_kernel_id<TestKernel1>();

  // expected-error@+1  {{no matching member function for call to 'has_kernel'}}
  kbSrc.has_kernel<TestKernel1>();

  // expected-error@+1  {{no matching member function for call to 'has_kernel'}}
  kbSrc.has_kernel<TestKernel1>(devices[0]);

  // expected-error@+1  {{no matching member function for call to 'has_kernel'}}
  kbSrc.has_kernel(TestKernel1ID);

  // expected-error@+1  {{no matching member function for call to 'has_kernel'}}
  kbSrc.has_kernel(TestKernel1ID, devices[0]);

  // expected-error@+1  {{no matching member function for call to 'begin'}}
  kbSrc.begin();

  // expected-error@+1  {{no matching member function for call to 'end'}}
  kbSrc.end();

  // expected-error@+1  {{no matching member function for call to 'empty'}}
  kbSrc.empty();

  std::string log;
  std::vector<std::string> flags{"-cl-fast-relaxed-math",
                                 "-cl-finite-math-only"};
  // OK
  syclex::build(kbSrc);

  // expected-error@+1 {{no matching function for call to 'build'}}
  syclex::build(kbSrc,
                syclex::properties{syclex::usm_kind<sycl::usm::alloc::host>});

  // OK
  syclex::build(kbSrc, syclex::properties{syclex::build_options{flags},
                                          syclex::save_log{&log}});

  // expected-error@+1 {{no matching function for call to 'build'}}
  syclex::build(kbSrc, syclex::properties{
                           syclex::build_options{flags}, syclex::save_log{&log},
                           syclex::usm_kind<sycl::usm::alloc::host>});
  // OK
  syclex::build(kbSrc, devices);

  // expected-error@+1 {{no matching function for call to 'build'}}
  syclex::build(kbSrc, devices,
                syclex::properties{syclex::usm_kind<sycl::usm::alloc::host>});

  // OK
  syclex::build(
      kbSrc, devices,
      syclex::properties{syclex::build_options{flags}, syclex::save_log{&log}});

  // expected-error@+1 {{no matching function for call to 'build'}}
  syclex::build(kbSrc, devices,
                syclex::properties{syclex::build_options{flags},
                                   syclex::save_log{&log},
                                   syclex::usm_kind<sycl::usm::alloc::host>});

#endif
}
