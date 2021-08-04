#include "non_cached_kernel_lock.hpp"

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

NonCachedKernelLock::~NonCachedKernelLock() {}

Locked<RT::PiKernel> NonCachedKernelLock::lockKernel(RT::PiKernel &K) {
  std::mutex &Mtx = Map[K];

  return {K, Mtx};
}

// static
bool NonCachedKernelLock::Compare(RT::PiKernel Lhs, RT::PiKernel Rhs) {
  return Lhs < Rhs;
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
