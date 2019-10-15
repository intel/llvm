#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>

namespace cl {
namespace sycl {
namespace detail {

AccessorImplHost::~AccessorImplHost() {
  if (MBlockedCmd)
    detail::Scheduler::getInstance().releaseHostAccessor(this);
}
}
}
}

