#include <detail/global_handler.hpp>

#include <mutex>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
GlobalHandler *SyclGlobalObjectsHandler;
std::mutex GlobalWritesAllowed;

GlobalHandler &GlobalHandler::instance() {
  if (!SyclGlobalObjectsHandler) {
    const std::lock_guard<std::mutex> Lock{GlobalWritesAllowed};
    if (!SyclGlobalObjectsHandler) {
      SyclGlobalObjectsHandler = new GlobalHandler();
    }
  }

  return *SyclGlobalObjectsHandler;
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
