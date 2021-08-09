#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/locked.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <map>
#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class NonCachedKernelLock {
public:
  ~NonCachedKernelLock();

  Locked<RT::PiKernel> lockKernel(RT::PiKernel &K);

private:
  using KernelLockMapT = std::map<RT::PiKernel, std::mutex>;

  std::mutex MapMtx;
  KernelLockMapT Map{std::less<RT::PiKernel>{}};
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
