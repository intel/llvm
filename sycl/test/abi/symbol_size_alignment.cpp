// RUN: %clangxx -fsycl %s -o %t
// UNSUPPORTED: libcxx

// Changing symbol size or alignment is a breaking change. If it happens, refer
// to the ABI Policy Guide for further instructions on breaking ABI.

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/image_impl.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_event.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/image.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/multi_ptr.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/program.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/sampler.hpp>
#include <CL/sycl/stream.hpp>
#include <CL/sycl/types.hpp>

using namespace cl::sycl;

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
  using accessor_t = accessor<int, 1, access::mode::read,
                              access::target::global_buffer,
                              access::placeholder::true_t>;
  check<accessor_t, 32, 8>();
  check<detail::AccessorImplDevice<1>, 24, 8>();
  check<detail::LocalAccessorBaseDevice<1>, 24, 8>();
  check<detail::AccessorImplHost, 128, 8>();
  check<detail::AccessorBaseHost, 16, 8>();
  check<detail::LocalAccessorImplHost, 56, 8>();
  check<buffer<int>, 40, 8>();
  check<context, 16, 8>();
  check<cpu_selector, 8, 8>();
  check<device, 16, 8>();
  check<device_event, 8, 8>();
  check<device_selector, 8, 8>();
  check<event, 16, 8>();
  check<gpu_selector, 8, 8>();
#ifdef _MSC_VER
  check<handler, 552, 8>();
  check<detail::buffer_impl, 216, 8>();
  check<detail::image_impl<1>, 272, 8>();
#else
  check<handler, 560, 8>();
  check<detail::buffer_impl, 184, 8>();
  check<detail::image_impl<1>, 240, 8>();
#endif
  check<image<1>, 16, 8>();
  check<kernel, 16, 8>();
  check<platform, 16, 8>();
#ifdef __SYCL_DEVICE_ONLY__
  check<private_memory<int, 1>, 4, 4>();
  check<detail::sampler_impl, 8, 8>();
#endif
  check<program, 16, 8>();
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
