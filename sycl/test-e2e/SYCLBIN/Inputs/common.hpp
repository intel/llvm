#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/syclbin_kernel_bundle.hpp>
#include <sycl/kernel_bundle.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclext = sycl::ext::oneapi;

#if !defined(SYCLBIN_INPUT_STATE) && !defined(SYCLBIN_OBJECT_STATE) &&         \
    !defined(SYCLBIN_EXECUTABLE_STATE)
#error "SYCLBIN state define missing!"
#endif

template <sycl::bundle_state InvalidState>
constexpr std::string_view GetStateName() {
  if constexpr (InvalidState == sycl::bundle_state::input)
    return "input";
  else if constexpr (InvalidState == sycl::bundle_state::object)
    return "object";
  else
    return "executable";
}

template <sycl::bundle_state InvalidState>
int ExpectExceptionInvalidState(const sycl::context &Ctx, const char *File) {
  try {
    syclexp::get_kernel_bundle<InvalidState>(Ctx, std::string{File});
    std::cout << "Unexpectedly created a kernel bundle for invalid state: "
              << GetStateName<InvalidState>() << std::endl;
    return 1;
  } catch (sycl::exception &) {
  }
  return 0;
}

// SYCLBIN is only directly loadable in the state they were produced in, so
// we run checks to ensure other states will complain.
int CommonLoadCheck(const sycl::context &Ctx, const char *File) {
  int Failed = 0;

#ifndef SYCLBIN_INPUT_STATE
  Failed += ExpectExceptionInvalidState<sycl::bundle_state::input>(Ctx, File);
#endif
#ifndef SYCLBIN_OBJECT_STATE
  Failed += ExpectExceptionInvalidState<sycl::bundle_state::object>(Ctx, File);
#endif
#ifndef SYCLBIN_EXECUTABLE_STATE
  Failed +=
      ExpectExceptionInvalidState<sycl::bundle_state::executable>(Ctx, File);
#endif

  return Failed;
}
