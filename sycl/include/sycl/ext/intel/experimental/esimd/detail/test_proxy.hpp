//==-------------- test_proxy.hpp - DPC++ Explicit SIMD API ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test proxy to differentiate move and copy constructor calls
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

// The test proxy if solely for the test purposes, so it's off by default, with
// no any code generated. It is enabled only if the __ESIMD_ENABLE_TEST_PROXY
// macro is defined.
// It's expected for the proxy class to be available in device code, so that it
// could be incorporated into the ESIMD API classes. Though there is no reason
// to limit it to the __SYCL_DEVICE_ONLY__.
#ifndef __ESIMD_ENABLE_TEST_PROXY

// No code generation by default
#define __ESIMD_DECLARE_TEST_PROXY
#define __ESIMD_DECLARE_TEST_PROXY_ACCESS
#define __esimd_move_test_proxy(other)

#else

// Declare the class attribute
//
// We are using non static data member initialization approach to force
// the value required. Initialization will take place even if no
// default/copy/move constructor of the test_proxy class was explcitly
// called by any of the user-defined constructors of the proxy target
#define __ESIMD_DECLARE_TEST_PROXY                                             \
  esimd::detail::test::test_proxy M_testProxy =                                \
      esimd::detail::test::test_proxy();

// Declare the getter to access the proxy from the tests
#define __ESIMD_DECLARE_TEST_PROXY_ACCESS                                      \
  const auto &get_test_proxy() const { return M_testProxy; }

// Test proxy will be handled in a proper way by default/implicit move
// constructors and move operators.
// Still the user-defined constructors or move operators should explicitly state
// what to do with each of class atributes, so a proper wrapper required
//
// We are using a simple do-while trick to make sure no code breakage could
// possibly occur in case macro becomes multistatement (PRE10-C in SEI CERT C)
#define __esimd_move_test_proxy(other)                                         \
  do {                                                                         \
    M_testProxy = std::move(other.M_testProxy);                                \
  } while (false)

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl::ext::intel::experimental::esimd::detail::test {

// The test_proxy class.
// Being intended solely for the test purposes, it is enabled only if the
// __ESIMD_ENABLE_TEST_PROXY macro is defined, which is off by default.
//
// This is a helper class for tests to differentiate between the copy
// constructor/assignment and the move constructor/assignment calls,
// as the copy constructor works as the default fallback for every case with
// move constructor disabled or not provided
//
// It is expected for the class with the test proxy (the class under test) to:
// - provide the get_test_proxy() method
// - properly handle moving the test_proxy member in user-defined move
//   constructors and user-defined assignment operators
//
// Therefore the following expression is expected to return `true` only if the
// move constructor or move operator was called for the instance of the class
// under test:
//   instance.get_test_proxy().was_move_destination()
//
class test_proxy {
  // Define the default value to use for every constructor
  bool M_move_destination = false;

public:
  test_proxy() { __esimd_dbg_print(test_proxy()); }

  test_proxy(const test_proxy &) {
    __esimd_dbg_print(test_proxy(const test_proxy &other));
  }
  test_proxy(test_proxy &&) {
    __esimd_dbg_print(test_proxy(test_proxy && other));
    M_move_destination = true;
  }
  test_proxy &operator=(const test_proxy &) {
    __esimd_dbg_print(test_proxy::operator=(const test_proxy &other));
    return *this;
  }
  test_proxy &operator=(test_proxy &&) {
    __esimd_dbg_print(test_proxy::operator=(test_proxy &&other));
    M_move_destination = true;
    return *this;
  }
  bool was_move_destination() const { return M_move_destination; }
};

} // namespace sycl::ext::intel::experimental::esimd::detail::test
} // __SYCL_INLINE_NAMESPACE(cl)

#endif // __ESIMD_ENABLE_TEST_PROXY

/// @endcond ESIMD_DETAIL
