// XFAIL: any-device-is-opencl, any-device-is-cuda, any-device-is-level_zero, gpu-intel-dg2, hip_amd
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15819
// RUN: %if any-device-is-opencl %{ %{build} -o %t-opencl.out %}
// RUN: %if any-device-is-level_zero %{ %{build} -DBUILD_FOR_L0 -o %t-l0.out %}
// RUN: %if any-device-is-cuda %{ %{build} -DBUILD_FOR_CUDA -o %t-cuda.out %}
// RUN: %if any-device-is-hip %{ %{build} -DBUILD_FOR_HIP -o %t-hip.out %}

#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>
using namespace sycl;

#ifdef BUILD_FOR_CUDA
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
constexpr auto BACKEND = backend::ext_oneapi_cuda;
using nativeDevice = CUdevice;
using nativeQueue = CUstream;
using nativeEvent = CUevent;
#elif defined(BUILD_FOR_HIP)
#include <sycl/ext/oneapi/backend/hip.hpp>
constexpr auto BACKEND = backend::ext_oneapi_hip;
using nativeDevice = hipDevice_t;
using nativeQueue = hipStream_t;
using nativeEvent = hipEvent_t;
#elif defined(BUILD_FOR_L0)
constexpr auto BACKEND = backend::ext_oneapi_level_zero;
using nativeDevice = ze_device_handle_t;
using nativeQueue = ze_command_queue_handle_t;
using nativeEvent = ze_event_handle_t;
#else
constexpr auto BACKEND = backend::opencl;
using nativeDevice = cl_device;
using nativeQueue = cl_command_queue;
using nativeEvent = cl_event;
#endif

constexpr int N = 100;
constexpr int VAL = 3;

int main() {

  assert(static_cast<bool>(
      std::is_same_v<backend_traits<BACKEND>::return_type<device>,
                     nativeDevice>));
  assert(static_cast<bool>(
      std::is_same_v<backend_traits<BACKEND>::return_type<queue>,
                     nativeQueue>));
  assert(static_cast<bool>(
      std::is_same_v<backend_traits<BACKEND>::return_type<event>,
                     nativeEvent>));

  device Device;
  backend_traits<BACKEND>::return_type<device> NativeDevice =
      get_native<BACKEND>(Device);
  // Create sycl device with a native device.
  auto InteropDevice = make_device<BACKEND>(NativeDevice);

  context Context(InteropDevice);

  // Create sycl queue with device created from a native device.
  queue Queue(InteropDevice, {sycl::property::queue::in_order()});
  backend_traits<BACKEND>::return_type<queue> NativeQueue =
      get_native<BACKEND>(Queue);
  auto InteropQueue = make_queue<BACKEND>(NativeQueue, Context);

  auto A = (int *)malloc_device(N * sizeof(int), InteropQueue);
  std::vector<int> vec(N, 0);

  auto Event = Queue.submit([&](handler &h) {
    h.parallel_for<class kern1>(range<1>(N),
                                [=](id<1> item) { A[item] = VAL; });
  });

  backend_traits<BACKEND>::return_type<event> NativeEvent =
      get_native<BACKEND>(Event);
  // Create sycl event with a native event.
  event InteropEvent = make_event<BACKEND>(NativeEvent, Context);

  // depends_on sycl event created from a native event.
  auto Event2 = InteropQueue.submit([&](handler &h) {
    h.depends_on(InteropEvent);
    h.parallel_for<class kern2>(range<1>(N), [=](id<1> item) { A[item]++; });
  });

  auto Event3 = InteropQueue.memcpy(&vec[0], A, N * sizeof(int), Event2);
  Event3.wait();

  if constexpr (BACKEND == backend::ext_oneapi_hip) {
    try {
      backend_traits<BACKEND>::return_type<context> NativeContext =
            get_native<BACKEND>(Context);
    } catch (sycl::exception &e) {
      assert(e.code() == sycl::errc::feature_not_supported);
    }
  }

  free(A, InteropQueue);

  for (const auto &val : vec) {
    assert(val == VAL + 1);
  }

  return 0;
}
