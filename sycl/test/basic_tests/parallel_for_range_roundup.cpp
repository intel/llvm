// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

using namespace sycl;

class PF_init_item;
class PF_init_id;
class PF_init_id1;

struct SizesInfo {
  range<1> ItemGlobalSize = {0};
  range<1> RealGlobalSizeX = {0};
  range<1> RealGlobalSizeY = {0};
  range<1> RealGlobalSizeZ = {0};
};

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

int try_item(size_t size) {
  size_t RoundedUpSize = (size + 32 - 1) / 32 * 32;
  SizesInfo SInfo;
  range<1> Size{size};
  int Counter = 0;
  bool OnGpu;

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, Size);
    queue myQueue;
    OnGpu = myQueue.get_device().is_gpu();

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);

      cgh.parallel_for<PF_init_item>(Size, [=](item<1> ITEM) {
        // cgh.parallel_for<PF_init_item>(Size, [=](int ITEM) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ITEM.get_range(0);
#ifdef __SYCL_DEVICE_ONLY__
        AccSizes[0].RealGlobalSizeX = {__spirv_GlobalSize_x()};
        AccSizes[0].RealGlobalSizeY = {__spirv_GlobalSize_y()};
        AccSizes[0].RealGlobalSizeZ = {__spirv_GlobalSize_z()};
#endif // __SYCL_DEVICE_ONLY__
      });
    });
  }

  std::cout << std::endl;
  if (OnGpu) {
    std::cout << "Ran on GPU" << std::endl;
    check("Real global size X = ", SInfo.RealGlobalSizeX.get(0), RoundedUpSize);
    check("Real global size Y = ", SInfo.RealGlobalSizeY.get(0), 1);
    check("Real global size Z = ", SInfo.RealGlobalSizeZ.get(0), 1);
  }
  check("Size seen by user = ", SInfo.ItemGlobalSize.get(0), size);
  check("Counter = ", Counter, size);
  std::cout << std::endl;
  return 0;
}

int try_id(size_t size) {
  size_t RoundedUpSize = (size + 32 - 1) / 32 * 32;
  SizesInfo SInfo;
  range<1> Size{size};
  int Counter = 0;
  bool OnGpu;

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, Size);
    queue myQueue;
    OnGpu = myQueue.get_device().is_gpu();

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);

      cgh.parallel_for<class PF_init_id>(Size, [=](id<1> ID) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ID[0];
#ifdef __SYCL_DEVICE_ONLY__
        AccSizes[0].RealGlobalSizeX = {__spirv_GlobalSize_x()};
        AccSizes[0].RealGlobalSizeY = {__spirv_GlobalSize_y()};
        AccSizes[0].RealGlobalSizeZ = {__spirv_GlobalSize_z()};
#endif // __SYCL_DEVICE_ONLY__
      });
    });
  }
  std::cout << std::endl;
  if (OnGpu) {
    std::cout << "Ran on GPU" << std::endl;
    check("Real global size X = ", SInfo.RealGlobalSizeX.get(0), RoundedUpSize);
    check("Real global size Y = ", SInfo.RealGlobalSizeY.get(0), 1);
    check("Real global size Z = ", SInfo.RealGlobalSizeZ.get(0), 1);
  }
  check("Counter = ", Counter, size);
  std::cout << std::endl;

  {
    buffer<SizesInfo, 1> BufSizes(&SInfo, 1);
    buffer<int, 1> BufCounter(&Counter, Size);
    queue myQueue;
    Counter = 0;
    OnGpu = myQueue.get_device().is_gpu();

    myQueue.submit([&](handler &cgh) {
      auto AccSizes = BufSizes.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_id1>(Size, [=](id<1> ID) {
        AccCounter[0].fetch_add(1);
        AccSizes[0].ItemGlobalSize = ID[0];
#ifdef __SYCL_DEVICE_ONLY__
        AccSizes[0].RealGlobalSizeX = {__spirv_GlobalSize_x()};
        AccSizes[0].RealGlobalSizeY = {__spirv_GlobalSize_y()};
        AccSizes[0].RealGlobalSizeZ = {__spirv_GlobalSize_z()};
#endif // __SYCL_DEVICE_ONLY__
      });
    });
  }
  std::cout << std::endl;
  if (OnGpu) {
    std::cout << "Ran on GPU" << std::endl;
    check("Real global size X = ", SInfo.RealGlobalSizeX.get(0), RoundedUpSize);
    check("Real global size Y = ", SInfo.RealGlobalSizeY.get(0), 1);
    check("Real global size Z = ", SInfo.RealGlobalSizeZ.get(0), 1);
  }
  check("Counter = ", Counter, size);
  std::cout << std::endl;

  return 0;
}

int main() {
  int x;

  x = 10;
  try_item(x);
  try_id(x);
  x = 256;
  try_item(x);
  try_id(x);

  return 0;
}