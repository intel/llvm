#include <atomic>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class SpinLock {
public:
  void lock() {
    while (MLock.exchange(true, std::memory_order_acquire));
  }
  void unlock() {
    MLock.store(false, std::memory_order_release);
  }

private:
  std::atomic_bool MLock;
};
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
