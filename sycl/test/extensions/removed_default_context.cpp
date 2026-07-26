// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -Xclang -verify %s
// REQUIRES: preview-breaking-changes-supported

// expected-no-diagnostics

#include <sycl/sycl.hpp>

#include <type_traits>
#include <utility>

#ifdef SYCL_EXT_ONEAPI_DEFAULT_CONTEXT
#error "SYCL_EXT_ONEAPI_DEFAULT_CONTEXT must not be defined"
#endif

#ifndef SYCL_EXT_ONEAPI_DEVICE_DEFAULT_CONTEXT
#error "SYCL_EXT_ONEAPI_DEVICE_DEFAULT_CONTEXT must remain defined"
#endif

template <typename T, typename = void>
struct has_ext_oneapi_get_default_context : std::false_type {};

template <typename T>
struct has_ext_oneapi_get_default_context<
    T, std::void_t<decltype(
           std::declval<T>().ext_oneapi_get_default_context())>>
    : std::true_type {};

static_assert(!has_ext_oneapi_get_default_context<sycl::platform>::value);
static_assert(has_ext_oneapi_get_default_context<sycl::device>::value);
static_assert(std::is_same_v<
              decltype(std::declval<sycl::platform>().khr_get_default_context()),
              sycl::context>);
