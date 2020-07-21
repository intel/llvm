// RUN: %clangxx -fsycl %s -o %t

// Changing symbol size is a breaking change. If it happens, refer to the ABI
// Policy Guide for further instructions on breaking ABI.

#include <CL/sycl/accessor.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/detail/accessor_impl.hpp>
#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/image_impl.hpp>
#include <CL/sycl/detail/stream_impl.hpp>
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

using namespace cl::sycl;

template <int newSize, int oldSize>
void check_size() {
  static_assert(newSize == oldSize, "Symbol size has changed.");
}

template <typename T, size_t oldSize>
void check_size() {
  check_size<sizeof(T), oldSize>();
}

int main() {
  using accessor_t = accessor<int, 1, access::mode::read,
                              access::target::global_buffer,
                              access::placeholder::true_t>;
  check_size<accessor_t, 32>();
  check_size<detail::AccessorImplDevice<1>, 24>();
  check_size<detail::LocalAccessorBaseDevice<1>, 24>();
  check_size<detail::AccessorImplHost, 128>();
  check_size<detail::AccessorBaseHost, 16>();
  check_size<detail::LocalAccessorImplHost, 56>();
  check_size<buffer<int>, 40>();
  check_size<context, 16>();
  check_size<cpu_selector, 8>();
  check_size<device, 16>();
  check_size<device_event, 8>();
  check_size<device_selector, 8>();
  check_size<event, 16>();
  check_size<gpu_selector, 8>();
#ifdef _MSC_VER
  check_size<handler, 552>();
  check_size<detail::buffer_impl, 288>();
  check_size<detail::image_impl<1>, 344>();
#else
  check_size<handler, 560>();
  check_size<detail::buffer_impl, 256>();
  check_size<detail::image_impl<1>, 312>();
#endif
  check_size<image<1>, 16>();
  check_size<kernel, 16>();
  check_size<platform, 16>();
#ifdef __SYCL_DEVICE_ONLY__
  check_size<private_memory<int, 1>, 4>();
  check_size<detail::sampler_impl, 8>();
#endif
  check_size<program, 16>();
  check_size<range<1>, 8>();
  check_size<sampler, 16>();
  check_size<stream, 144>();
  check_size<detail::stream_impl, 120>();
  check_size<queue, 16>();

  return 0;
}
