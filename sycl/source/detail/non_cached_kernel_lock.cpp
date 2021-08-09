#include "non_cached_kernel_lock.hpp"

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

NonCachedKernelLock::~NonCachedKernelLock() {}

Locked<RT::PiKernel> NonCachedKernelLock::lockKernel(RT::PiKernel &K) {
  std::unique_lock<std::mutex> MapLock{MapMtx};
  std::mutex &Mtx = Map[K];
  MapLock.unlock();

  return {K, Mtx};
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
