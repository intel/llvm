#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/util.hpp>
#include <detail/platform_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
using PlatformImplPtr = std::shared_ptr<platform_impl>;

struct GlobalHandler;

extern GlobalHandler *SyclGlobalObjectsHandler;

/// Wrapper class for global data structures with non-trivial destructors.
struct GlobalHandler {
  Scheduler IScheduler;
  ProgramManager IProgramManager;
  Sync ISync;
  std::vector<PlatformImplPtr> IPlatformCache;
  std::mutex IPlatformMapMutex;
  std::mutex IFilterMutex;

  static GlobalHandler &instance();

  GlobalHandler(const GlobalHandler &) = delete;
  GlobalHandler(GlobalHandler &&) = delete;

private:
  GlobalHandler() = default;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
