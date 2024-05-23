// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_2(int *ptr, int start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + start;
}

// Templated free function definition.
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    ext::oneapi::experimental::single_task_kernel)) void ff_3(T *ptr, T start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + start;
}

// Explicit instantiation of free function.
template void ff_3(int *ptr, int start);

// A plain function that is not a free function.
void ff_4(int *ptr, int start) {}

bool test_kernel_apis(queue Queue) {
  bool Pass = true;

#ifndef __SYCL_DEVICE_ONLY__
  // Check for a free function, which is known to be a free function.
  // Expect: true.
  bool Pass1 = ext::oneapi::experimental::is_nd_range_kernel_v<ff_2, 2>;
  std::cerr << "Pass1=" << Pass1 << std::endl;
  Pass &= Pass1;

  // Check for a free function which in not a free function.
  // Expect: false.
  bool Pass2 = !ext::oneapi::experimental::is_nd_range_kernel_v<ff_4, 1>;
  std::cerr << "Pass2=" << Pass2 << std::endl;
  Pass &= Pass2;

  // Check for a free function, which has different Dim.
  // Expect: false.
  bool Pass3 = !ext::oneapi::experimental::is_nd_range_kernel_v<ff_2, 1>;
  std::cerr << "Pass3=" << Pass3 << std::endl;
  Pass &= Pass3;

  // Check for a single-task free function, which is a free function.
  // Expect: true.
  bool Pass4 = ext::oneapi::experimental::is_single_task_kernel_v<(
      void (*)(int *, int))ff_3>;
  std::cerr << "Pass4=" << Pass4 << std::endl;
  Pass &= Pass4;

  // Check for a single-task free function, which is a not free function.
  // Expect: false.
  bool Pass5 = !ext::oneapi::experimental::is_single_task_kernel_v<ff_4>;
  std::cerr << "Pass5=" << Pass5 << std::endl;
  Pass &= Pass5;

  // Check for a free function.
  // Expect: true.
  bool Pass6 = ext::oneapi::experimental::is_kernel_v<ff_2>;
  std::cerr << "Pass6=" << Pass6 << std::endl;
  Pass &= Pass6;

  // Check for a free function.
  // Expect: true.
  bool Pass7 =
      ext::oneapi::experimental::is_kernel_v<(void (*)(int *, int))ff_3>;
  std::cerr << "Pass7=" << Pass7 << std::endl;
  Pass &= Pass7;

  // Check for a free function which is not a free function.
  // Expect false.
  bool Pass8 = !ext::oneapi::experimental::is_kernel_v<ff_4>;
  std::cerr << "Pass8=" << Pass8 << std::endl;
  Pass &= Pass8;

  std::cout << "Test kernel APIs: " << (Pass ? "PASS" : "FAIL") << std::endl;
#endif
  return Pass;
}

bool test_bundle_apis(queue Queue) {
  bool Pass = true;

#ifndef __SYCL_DEVICE_ONLY__
  context Context{Queue.get_context()};
  device Device{Queue.get_device()};
  std::vector<device> Devices{Context.get_devices()};

  // ff_2 and ff_3 are free functions, evaluate has_kernel_bundle.
  bool PassA = has_kernel_bundle<ff_2, bundle_state::executable>(Context);
  std::cerr << "PassA=" << PassA << std::endl;
  Pass &= PassA;

  bool PassB =
      has_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context);
  std::cerr << "PassB=" << PassB << std::endl;
  Pass &= PassB;

  // ff_2 and ff_3 are free functions, evaluate has_kernel_bundle.
  bool PassC =
      has_kernel_bundle<ff_2, bundle_state::executable>(Context, Devices);
  std::cerr << "PassC=" << PassC << std::endl;
  Pass &= PassC;

  bool PassD =
      has_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context, Devices);
  std::cerr << "PassD=" << PassD << std::endl;
  Pass &= PassD;

  // ff_3 is compatible.
  bool PassE = is_compatible<(void (*)(int *, int))ff_3>(Device);
  std::cerr << "PassE=" << PassE << std::endl;
  Pass &= PassE;

  // Check that ff_2 is found in bundle.
  kernel_bundle Bundle2 =
      get_kernel_bundle<ff_2, bundle_state::executable>(Context);
  bool PassF = Bundle2.ext_oneapi_has_kernel<ff_2>();
  std::cerr << "PassF=" << PassF << std::endl;
  Pass &= PassF;

  bool PassG = Bundle2.ext_oneapi_has_kernel<ff_2>(Device);
  std::cerr << "PassG=" << PassG << std::endl;
  Pass &= PassG;
  kernel Kernel2 = Bundle2.ext_oneapi_get_kernel<ff_2>();
  bool PassH = true;
  std::cerr << "PassH=" << PassH << std::endl;
  Pass &= PassH;

  // Check that ff_3 is found in bundle.
  kernel_bundle Bundle3 =
      get_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context);
  bool PassI = Bundle3.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>();
  std::cerr << "PassI=" << PassI << std::endl;
  Pass &= PassI;

  bool PassJ =
      Bundle3.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>(Device);
  std::cerr << "PassJ=" << PassJ << std::endl;
  Pass &= PassJ;
  kernel Kernel3 = Bundle3.ext_oneapi_get_kernel<(void (*)(int *, int))ff_3>();
  bool PassK = true;
  std::cerr << "PassK=" << PassK << std::endl;
  Pass &= PassK;

  kernel_bundle Bundle4 =
      get_kernel_bundle<bundle_state::executable>(Context, Devices);
  kernel_id Id = get_kernel_id<ff_2>();
  bool PassL = true;
  std::cerr << "PassL=" << PassL << std::endl;
  Pass &= PassL;

  // Check that ff_2 is found in bundle obtained using devices.
  kernel_bundle Bundle5 =
      get_kernel_bundle<ff_2, bundle_state::executable>(Context, Devices);
  bool PassM = Bundle5.ext_oneapi_has_kernel<ff_2>();
  std::cerr << "PassM=" << PassM << std::endl;
  Pass &= PassM;

  bool PassN = Bundle5.ext_oneapi_has_kernel<ff_2>(Device);
  std::cerr << "PassN=" << PassN << std::endl;
  Pass &= PassN;

  kernel Kernel5 = Bundle5.ext_oneapi_get_kernel<ff_2>();
  bool PassO = true;
  std::cerr << "PassO=" << PassO << std::endl;
  Pass &= PassO;

  // Check that ff_3 is found in bundle obtained using devices.
  kernel_bundle Bundle6 =
      get_kernel_bundle<(void (*)(int *, int))ff_3, bundle_state::executable>(
          Context, Devices);
  bool PassP = Bundle6.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>();
  std::cerr << "PassP=" << PassP << std::endl;
  Pass &= PassP;

  bool PassQ =
      Bundle6.ext_oneapi_has_kernel<(void (*)(int *, int))ff_3>(Device);
  std::cerr << "PassQ=" << PassQ << std::endl;
  Pass &= PassQ;

  kernel Kernel6 = Bundle6.ext_oneapi_get_kernel<(void (*)(int *, int))ff_3>();
  bool PassR = true;
  std::cerr << "PassR=" << PassR << std::endl;
  Pass &= PassR;
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
