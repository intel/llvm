// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
//
// Checks that the khr compile, link and build overloads reject khr properties
// that are not compile, link or build properties.

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS
#include <sycl/sycl.hpp>

#include <vector>

namespace kp = sycl::khr::property;
using namespace sycl::khr;

using input_bundle = sycl::kernel_bundle<sycl::bundle_state::input>;
using object_bundle = sycl::kernel_bundle<sycl::bundle_state::object>;

void compile_foreign(const input_bundle &ib,
                     const std::vector<sycl::device> &devs) {
  // expected-error@+1 {{no matching function for call to 'compile'}}
  auto o1 = sycl::compile(ib, devs, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'compile'}}
  auto o2 = sycl::compile(ib, properties{kp::enable_profiling{}});
}

void link_foreign(const object_bundle &ob,
                  const std::vector<object_bundle> &obs,
                  const std::vector<sycl::device> &devs) {
  // expected-error@+1 {{no matching function for call to 'link'}}
  auto e1 = sycl::link(obs, devs, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'link'}}
  auto e2 = sycl::link(ob, devs, properties{kp::in_order{}});
  // expected-error@+1 {{no matching function for call to 'link'}}
  auto e3 = sycl::link(obs, kp::enable_profiling{});
  // expected-error@+1 {{no matching function for call to 'link'}}
  auto e4 = sycl::link(ob, properties{kp::enable_profiling{}});
}

void build_foreign(const input_bundle &ib,
                   const std::vector<sycl::device> &devs) {
  // expected-error@+1 {{no matching function for call to 'build'}}
  auto e1 = sycl::build(ib, devs, kp::in_order{});
  // expected-error@+1 {{no matching function for call to 'build'}}
  auto e2 = sycl::build(ib, properties{kp::in_order{}, kp::enable_profiling{}});
}
