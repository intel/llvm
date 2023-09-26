// RUN: %{build} %if ext_oneapi_hip %{ -DSYCL_EXT_ONEAPI_BACKEND_HIP %} %else %if ext_oneapi_cuda %{ -DSYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL %} %else %if ext_oneapi_level_zero %{ -DSYCL_EXT_ONEAPI_BACKEND_L0 %} -o test.out
// RUN: %{run} test.out

#include <sycl/sycl.hpp>
using namespace sycl;

#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA_EXPERIMENTAL
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>
constexpr auto BACKEND = backend::ext_oneapi_cuda;
#elif defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
#include <sycl/ext/oneapi/backend/hip.hpp>
constexpr auto BACKEND = backend::ext_oneapi_hip;
#elif defined(SYCL_EXT_ONEAPI_BACKEND_L0)
constexpr auto BACKEND = backend::ext_oneapi_level_zero;
#else
constexpr auto BACKEND = backend::opencl;
#endif

constexpr int N = 100;
constexpr int VAL = 3;

int main() {

  device Device;
  auto NativeDevice = get_native<BACKEND>(Device);
  // Create sycl device with a native device.
  auto InteropDevice = make_device<BACKEND>(NativeDevice);

  context Context(InteropDevice);

  // Create sycl queue with device created from a native device.
  queue Queue(InteropDevice, {sycl::property::queue::in_order()});
  auto NativeQueue = get_native<BACKEND>(Queue);
  auto InteropQueue = make_queue<BACKEND>(NativeQueue, Context);

  auto A = (int *)malloc_device(N * sizeof(int), InteropQueue);
  std::vector<int> vec(N, 0);

  auto Event = Queue.submit([&](handler &h) {
    h.parallel_for<class kern1>(range<1>(N),
                                [=](id<1> item) { A[item] = VAL; });
  });

  auto NativeEvent = get_native<BACKEND>(Event);
  // Create sycl event with a native event.
  event InteropEvent = make_event<BACKEND>(NativeEvent, Context);

  // depends_on sycl event created from a native event.
  auto Event2 = InteropQueue.submit([&](handler &h) {
    h.depends_on(InteropEvent);
    h.parallel_for<class kern2>(range<1>(N), [=](id<1> item) { A[item]++; });
  });

  auto Event3 = InteropQueue.memcpy(&vec[0], A, N * sizeof(int), Event2);
  Event3.wait();

  free(A, InteropQueue);

  for (const auto &val : vec) {
    assert(val == VAL + 1);
  }

  return 0;
}
