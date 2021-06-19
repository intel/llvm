#pragma once

#include "pi_args.hpp"
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/pi_api_id.hpp>

#include <functional>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace xpti_helpers {
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
      const typename as_function<                                              \
          void, typename PiApiArgTuple<detail::PiApiKind::api>::type>::type    \
          &Handler) {                                                          \
    MHandler##_##api = [Handler](void *Data) {                                 \
      typename PiApiArgTuple<detail::PiApiKind::api>::type TV{                 \
          static_cast<unsigned char *>(Data)};                                 \
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
