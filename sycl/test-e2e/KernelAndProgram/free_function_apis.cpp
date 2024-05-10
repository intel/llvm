#include <sycl/sycl.hpp>
#include <ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/kernel_bundle.hpp>
#include <iostream>

using namespace sycl;

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_2(int *ptr, int start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] =
      LId.get(0) + LId.get(1) + start;
}

// Templated free function definition.
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
  ext::oneapi::experimental::single_task_kernel)) void ff_3(T* ptr, T start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + start;
}

// Explicit instantiation of free function.
extern template void ff_3(int* ptr, int start);

// A plain function that is not a free function.
void ff_4(int* ptr, int start) {}

#if 1
bool test_kernel_apis(queue Queue)
{
  bool Pass = true;

  // Check for a free function, which is known to be a free function.
  // Expect: true.
  Pass &= ext::oneapi::experimental::is_nd_range_kernel_v<ff_2, 2>;

  // Check for a free function which in not a free function.
  // Expect: false.
  Pass &= !ext::oneapi::experimental::is_nd_range_kernel_v<ff_4, 1>;

  // Check for a free function, which has different Dim.
  // Expect: false.
  Pass &= !ext::oneapi::experimental::is_nd_range_kernel_v<ff_2, 1>;

  // Check for a single-task free function, which is a free function.
  // Expect: true.
  Pass &= ext::oneapi::experimental::is_single_task_kernel_v<(
      void (*)(int *, int))ff_3>;

  // Check for a single-task free function, which is a not free function.
  // Expect: false.
  Pass &= !ext::oneapi::experimental::is_single_task_kernel_v<ff_4>;

  // Check for a free function.
  // Expect: true.
  Pass &= ext::oneapi::experimental::is_kernel_v<ff_2>;

  // Check for a free function.
  // Expect: true.
  Pass &= ext::oneapi::experimental::is_kernel_v<(void (*)(int *, int))ff_3>;

  // Check for a free function which is not a free function.
  // Expect false.
  Pass &= !ext::oneapi::experimental::is_kernel_v<ff_4>;

  std::cout << "Test kernel APIs: " << (Pass ? "PASS" : "FAIL") << std::endl;
  return Pass;
}
#endif

bool test_bundle_apis(queue Queue) {
  bool Pass = true;
  context Context{Queue.get_context()};
  device Device{Queue.get_device()};
  std::vector<device> Devices{Context.get_devices()};
  std::cerr << "Num devices = " << Devices.size() << std::endl;

#if 1
  // ff_4 is not a free function, has_kernel_bundle will not be evaluated.
  if constexpr (ext::oneapi::experimental::is_kernel_v<ff_4>)
    Pass &= has_kernel_bundle<ff_4, bundle_state::executable>(Context);

  // ff_2 and ff_3 are free functions, evaluate has_kernel_bundle.
  Pass &= has_kernel_bundle<ff_2, bundle_state::executable>(Context);
#if 1
  if constexpr (ext::oneapi::experimental::is_kernel_v<(
    void (*)(int*, int))ff_3>)
  Pass &=
      has_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context);
#endif
  // ff_2 and ff_3 are free functions, evaluate has_kernel_bundle.
  Pass &= has_kernel_bundle<ff_2, bundle_state::executable>(Context, Devices);
#if 1
  if constexpr (ext::oneapi::experimental::is_kernel_v<(
    void (*)(int*, int))ff_3>)
  Pass &=
      has_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context, Devices);
#endif
#if 1
  // ff_3 is compatible.
  if constexpr (ext::oneapi::experimental::is_kernel_v<(
                    void (*)(int *, int))ff_3>)
    Pass &= is_compatible<(void (*)(int *, int))ff_3>(Device);
#endif
  // ff_4 is not compatible.
  if constexpr (ext::oneapi::experimental::is_kernel_v<ff_4>)
    Pass &= !is_compatible<ff_4>(Device);

  // Check that ff_2 is found in bundle.
  if constexpr (ext::oneapi::experimental::is_kernel_v<ff_2>) {
    kernel_bundle Bundle =
        get_kernel_bundle<ff_2, bundle_state::executable>(Context);
    Pass &= Bundle.ext_oneapi_has_kernel<ff_2>();
    Pass &= Bundle.ext_oneapi_has_kernel<ff_2>(Device);
    kernel Kernel = Bundle.ext_oneapi_get_kernel<ff_2>();
  }
#endif
#if 1
  // Check that ff_3 is found in bundle.
  if constexpr (ext::oneapi::experimental::is_kernel_v<(
                    void (*)(int *, int))ff_3>) {
    kernel_bundle Bundle =
        get_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
            Context);
    std::cerr << "Got bundle\n";
    Pass &= Bundle.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>();
    std::cerr << "Checked plain\n";
    Pass &= Bundle.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>(Device);
    std::cerr << "Checked with Devices\n";
    kernel Kernel = Bundle.ext_oneapi_get_kernel<(void (*)(int *, int))ff_3>();
    std::cerr << "Got kernel\n";
  }
#endif
  kernel_bundle Bundle = get_kernel_bundle<bundle_state::executable>(Context, Devices);
  std::cerr << "Got bundle\n";
  kernel_id Id = get_kernel_id<ff_2>();

#if 1
  // Check that ff_2 is found in bundle obtained using devices.
  if constexpr (ext::oneapi::experimental::is_kernel_v<ff_2>) {
    std::cerr << "Did constexpr\n";
#if 1
    kernel_bundle Bundle =
        get_kernel_bundle<ff_2, bundle_state::executable>(Context, Devices);
    std::cerr << "Got bundle\n";
    Pass &= Bundle.ext_oneapi_has_kernel<ff_2>();
    std::cerr << "Checked plain\n";
    Pass &= Bundle.ext_oneapi_has_kernel<ff_2>(Device);
    std::cerr << "Checked with Devices\n";
    kernel Kernel = Bundle.ext_oneapi_get_kernel<ff_2>();
    std::cerr << "Got kernel\n";
#endif
  }
#if 1
  // Check that ff_3 is found in bundle obtained using devices.
  if constexpr (ext::oneapi::experimental::is_kernel_v<(
                    void (*)(int *, int))ff_3>) {
    kernel_bundle Bundle =
        get_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
            Context, Devices);
    Pass &= Bundle.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>();
    Pass &= Bundle.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>(Device);
    kernel Kernel = Bundle.ext_oneapi_get_kernel<(void (*)(int *, int))ff_3>();
  }
#endif
  // Check that ff_4 is not checked for being in bundle.
  if constexpr (ext::oneapi::experimental::is_kernel_v<ff_4>) {
    kernel_bundle Bundle =
        get_kernel_bundle<ff_4, bundle_state::executable>(Context);
    Pass &= Bundle.ext_oneapi_has_kernel<ff_4>();
    Pass &= Bundle.ext_oneapi_has_kernel<ff_4>(Device);
    kernel Kernel = Bundle.ext_oneapi_get_kernel<ff_4>();
  }
#endif

  std::cout << "Test bundle APIs: " << (Pass ? "PASS" : "FAIL") << std::endl;
  return Pass;
}

int main() {
  queue Queue;

  bool Pass = true;
  Pass &= test_kernel_apis(Queue);
  Pass &= test_bundle_apis(Queue);

  return Pass ? 0 : 1;
}
