#include "detail/accessor_impl.hpp"
#include "detail/handler_impl.hpp"
#include <iostream>
#include <sycl/detail/native_cpu.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

NativeCPUArgDesc::NativeCPUArgDesc(const ArgDesc &Arg) {
  if (Arg.MType == kernel_param_kind_t::kind_accessor) {
    auto HostAcc = static_cast<AccessorImplHost *>(Arg.MPtr);
    MPtr = HostAcc->MData;
    MisAcc = true;
  } else {
    MPtr = Arg.MPtr;
    MisAcc = false;
  }
}

std::vector<NativeCPUArgDesc>
processArgsForNativeCPU(const std::vector<ArgDesc> &MArgs) {
  std::cout << "[PTRDBG] processing args\n";
  std::vector<NativeCPUArgDesc> res;
  for (auto &arg : MArgs) {
    res.emplace_back(arg);
  }
  return res;
}

void setNativeCPUImpl(std::shared_ptr<handler_impl> &MImpl,
                      std::shared_ptr<NativeCPUTask_t> &task) {
  MImpl->MNativeCPUFunct = task;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
