//==---------- pi_arguments_handler.hpp - PI call arguments handler --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/pi.hpp>

#include <functional>
#include <tuple>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace xpti_helpers {

template <typename TupleT, size_t... Is>
inline auto get(char *Data, const std::index_sequence<Is...> &) {
  // Our type should be last in Is sequence
  using TargetType =
      typename std::tuple_element<sizeof...(Is) - 1, TupleT>::type;

  // Calculate sizeof all elements before target + target element then substract
  // sizeof target element
  size_t Res = (sizeof(typename std::tuple_element<Is, TupleT>::type) + ...) -
               sizeof(TargetType);
  return *(typename std::decay<TargetType>::type *)(Data + Res);
}

template <typename TupleT, size_t... Is>
inline TupleT unpack(char *Data,
                     const std::index_sequence<Is...> & /*1..TupleSize*/) {
  return {get<TupleT>(Data, std::make_index_sequence<Is + 1>{})...};
}

/// PiArgumentsHandler is a helper class to process incoming XPTI function call
/// events and unpack contained arguments.
///
/// Usage:
///
/// PiArgumentsHandler provides set_<API name> member functions, that accept a
/// lambda with the same arguments as target PI API. Use it to set up handling
/// for particular API. By default an empty lambda is used.
///
/// When an event is signaled, use PiArgumentsHandler::handle() member function
/// to process the incoming event and call necessary handler.
///
/// See sycl/tools/pi-trace/ for an example.
class PiArgumentsHandler {
public:
  void handle(uint32_t ID, void *ArgsData) {
#define _PI_API(api, ...)                                                      \
  if (ID == static_cast<uint32_t>(detail::PiApiKind::api)) {                   \
    MHandler##_##api(ArgsData);                                                \
    return;                                                                    \
  }
#include <CL/sycl/detail/pi.def>
#undef _PI_API
  }

#define _PI_API(api, ...)                                                      \
  void set##_##api(std::function<void(__VA_ARGS__)> Handler) {                 \
    MHandler##_##api = [Handler](void *Data) {                                 \
      using TupleT = std::tuple<__VA_ARGS__>;                                  \
      TupleT Tuple = unpack<TupleT>(                                           \
          (char *)Data,                                                        \
          std::make_index_sequence<std::tuple_size<TupleT>::value>{});         \
      std::apply(Handler, Tuple);                                              \
    };                                                                         \
  }
#include <CL/sycl/detail/pi.def>
#undef _PI_API

private:
#define _PI_API(api, ...)                                                      \
  std::function<void(void *)> MHandler##_##api = [](void *) {};
#include <CL/sycl/detail/pi.def>
#undef _PI_API
};
} // namespace xpti_helpers
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
