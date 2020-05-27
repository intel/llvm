#include <CL/sycl/detail/handler_proxy.hpp>

#include <CL/sycl/handler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

void associateWithHandler(handler &CGH, AccessorBaseHost *Acc,
                          access::target Target) {
  CGH.associateWithHandler(Acc, Target);
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
