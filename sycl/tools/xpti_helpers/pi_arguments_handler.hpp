//==---------- pi_arguments_handler.hpp - PI call arguments handler --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/pi_api_id.hpp>
#include <CL/sycl/detail/pi_args_helper.hpp>

#include "tuple_view.hpp"

#include <functional>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace xpti_helpers {
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
#define _PI_API(api)                                                           \
  if (ID == sycl::detail::PiApiID<detail::PiApiKind::api>::id) {               \
    MHandler##_##api(ArgsData);                                                \
    return;                                                                    \
  }
#include <CL/sycl/detail/pi.def>
#undef _PI_API
  }

#define _PI_API(api)                                                           \
  void set##_##api(                                                            \
      typename as_function<void, typename detail::PiApiArgTuple<               \
                                     detail::PiApiKind::api>::type>::type      \
          Handler) {                                                           \
    MHandler##_##api = [Handler](void *Data) {                                 \
      typename as_tuple_view<                                                  \
          typename detail::PiApiArgTuple<detail::PiApiKind::api>::type>::type  \
          TV{static_cast<unsigned char *>(Data)};                              \
      xpti_helpers::apply(Handler, TV);                                        \
    };                                                                         \
  }
#include <CL/sycl/detail/pi.def>
#undef _PI_API

private:
#define _PI_API(api)                                                           \
  std::function<void(void *)> MHandler##_##api = [](void *) {};
#include <CL/sycl/detail/pi.def>
#undef _PI_API
};
} // namespace xpti_helpers
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
