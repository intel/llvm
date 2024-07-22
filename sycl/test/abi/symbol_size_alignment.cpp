// RUN: %clangxx -fsycl %s -D__SYCL_INTERNAL_API -o %t
// UNSUPPORTED: libcxx

// Changing symbol size or alignment is a breaking change. If it happens, refer
// to the ABI Policy Guide for further instructions on breaking ABI.

#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/device.hpp>
#include <sycl/device_event.hpp>
#include <sycl/device_selector.hpp>
#include <sycl/event.hpp>
#include <sycl/handler.hpp>
#include <sycl/image.hpp>
#include <sycl/kernel.hpp>
#include <sycl/multi_ptr.hpp>
#include <sycl/platform.hpp>
#include <sycl/queue.hpp>
#include <sycl/sampler.hpp>
#include <sycl/stream.hpp>
#include <sycl/types.hpp>

using namespace sycl;

template <int newSize, int oldSize> void check_size() {
  static_assert(newSize == oldSize, "Symbol size has changed.");
}

template <int newAlignment, int oldAlignment> void check_alignment() {
  static_assert(newAlignment == oldAlignment, "Alignment has changed");
}

template <typename T, size_t oldSize, size_t oldAlignment> void check() {
  check_size<sizeof(T), oldSize>();
  check_alignment<alignof(T), oldAlignment>();
}

int main() {
  using accessor_t =
      accessor<int, 1, access::mode::read, access::target::device,
               access::placeholder::true_t>;
  check<accessor_t, 32, 8>();
  check<detail::AccessorImplDevice<1>, 24, 8>();
  check<detail::LocalAccessorBaseDevice<1>, 24, 8>();
  check<detail::AccessorBaseHost, 16, 8>();
  check<buffer<int>, 40, 8>();
  check<context, 16, 8>();
  check<cpu_selector, 8, 8>();
  check<device, 16, 8>();
  check<device_event, 8, 8>();
  check<device_selector, 8, 8>();
  check<event, 16, 8>();
  check<gpu_selector, 8, 8>();
#ifdef _MSC_VER
  check<handler, 208, 8>();
#else
  check<handler, 216, 8>();
#endif
  check<image<1>, 16, 8>();
  check<kernel, 16, 8>();
  check<platform, 16, 8>();
#ifdef __SYCL_DEVICE_ONLY__
  check<private_memory<int, 1>, 4, 4>();
  check<detail::sampler_impl, 8, 8>();
#endif
  check<range<1>, 8, 8>();
  check<sampler, 16, 8>();
  check<stream, 144, 8>();
  check<queue, 16, 8>();
  check<vec<float, 1>, 4, 4>();
  check<vec<float, 2>, 8, 8>();
  check<vec<float, 4>, 16, 16>();
  check<vec<float, 8>, 32, 32>();
  check<vec<float, 16>, 64, 64>();
  check<vec<double, 1>, 8, 8>();
  check<vec<double, 2>, 16, 16>();
  check<vec<double, 4>, 32, 32>();
  check<vec<double, 8>, 64, 64>();

  return 0;
}
