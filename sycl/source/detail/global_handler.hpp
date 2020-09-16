#include <detail/program_manager/program_manager.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
struct GlobalHandler;

extern GlobalHandler *SyclGlobalObjectsHandler;

/// Wrapper class for global data structures with non-trivial destructors.
struct GlobalHandler {
  Scheduler scheduler;
  ProgramManager program_manager;

  static GlobalHandler &instance();

  GlobalHandler(const GlobalHandler &) = delete;
  GlobalHandler(GlobalHandler &&) = delete;

private:
  GlobalHandler() = default;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
