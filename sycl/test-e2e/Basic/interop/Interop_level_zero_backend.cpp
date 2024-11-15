// RUN: %if any-device-is-level_zero %{ %{build} -isystem %sycl_include -o %t-l0.out %}

#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>
using namespace sycl;

constexpr auto BACKEND = backend::ext_oneapi_level_zero;
using nativeDevice = ze_device_handle_t;
using nativeQueue = ze_command_queue_handle_t;
using nativeEvent = ze_event_handle_t;

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
  backend_traits<BACKEND>::input_type<queue> InputType(NativeQueue, Device);

  auto InteropQueue = make_queue<BACKEND>(InputType, Context);

  auto A = (int *)malloc_device(N * sizeof(int), InteropQueue);
  std::vector<int> vec(N, 0);

  auto Event = Queue.submit([&](handler &h) {
    h.parallel_for<class kern1>(range<1>(N),
                                [=](id<1> item) { A[item] = VAL; });
  });

  backend_traits<BACKEND>::return_type<event> NativeEvent =
      get_native<BACKEND>(Event);
  backend_traits<BACKEND>::input_type<event> EventInputType;
  EventInputType.NativeHandle = NativeEvent;
  // Create sycl event with a native event.
  event InteropEvent = make_event<BACKEND>(EventInputType, Context);

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
