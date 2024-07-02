// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda

// This test tests free function kernel code generation and execution.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

void printUSM(int *usmPtr, int size) {
  std::cout << "usmPtr[] = {";
  for (int i = 0; i < size; i++) {
    std::cout << usmPtr[i] << ", ";
  }
  std::cout << "}\n";
}

bool checkUSM(int *usmPtr, int size, int *Result) {
  bool Pass = true;
  for (int i = 0; i < size; i++) {
    if (usmPtr[i] != Result[i]) {
      Pass = false;
      break;
    }
  }
  if (Pass)
    return true;

  std::cout << "Expected = {";
  for (int i = 0; i < size; i++) {
    std::cout << Result[i] << ", ";
  }
  std::cout << "}\n";
  std::cout << "Result   = {";
  for (int i = 0; i < size; i++) {
    std::cout << usmPtr[i] << ", ";
  }
  std::cout << "}\n";
  return false;
}

extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel)) void ff_0(int *ptr,
                                                               int start,
                                                               int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + end;
}

bool test_0(queue Queue) {
  constexpr int Range = 10;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int start = 3;
  int end = 5;
  int Result[Range] = {0, 0, 0, 8, 8, 8, 0, 0, 0, 0};
  range<1> R1{Range};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.single_task([=]() {
      for (int i = start; i <= end; i++)
        usmPtr[i] = start + end;
    });
  });
  Queue.wait();
  bool PassA = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 0a: " << (PassA ? "PASS" : "FAIL") << std::endl;

  bool PassB = false;
#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  kernel_id Kernel_id = ext::oneapi::experimental::get_kernel_id<ff_0>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, start);
    Handler.set_arg(2, end);
    Handler.single_task(Kernel);
  });
  Queue.wait();
  PassB = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 0b: " << (PassB ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
#endif
  return PassA && PassB;
}

// Overloaded free function definition.
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void ff_1(int *ptr, int start, int end) {
  nd_item<1> Item = ext::oneapi::this_work_item::get_nd_item<1>();
  id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = GId.get(0) + start + end;
}

bool test_1(queue Queue) {
  constexpr int Range = 10;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int start = 3;
  int Result[Range] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
  nd_range<1> R1{{Range}, {1}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R1, [=](nd_item<1> Item) {
      id<1> GId = Item.get_global_id();
      usmPtr[GId.get(0)] = GId.get(0) + start + Range;
    });
  });
  Queue.wait();
  bool PassA = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 1a: " << (PassA ? "PASS" : "FAIL") << std::endl;

  bool PassB = false;
#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  kernel_id Kernel_id = ext::oneapi::experimental::get_kernel_id<(
      void (*)(int *, int, int))ff_1>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, start);
    Handler.set_arg(2, Range);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();
  PassB = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 1b: " << (PassB ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
#endif
  return PassA && PassB;
}

// Overloaded free function definition.
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_1(int *ptr, int start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + start;
}

bool test_2(queue Queue) {
  constexpr int Range = 16;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int value = 55;
  int Result[Range] = {55, 56, 55, 56, 56, 57, 56, 57,
                       55, 56, 55, 56, 56, 57, 56, 57};
  nd_range<2> R2{range<2>{4, 4}, range<2>{2, 2}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R2, [=](nd_item<2> Item) {
      int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(usmPtr);
      id<2> GId = Item.get_global_id();
      id<2> LId = Item.get_local_id();
      ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + value;
    });
  });
  Queue.wait();
  bool PassA = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 2a: " << (PassA ? "PASS" : "FAIL") << std::endl;

  bool PassB = false;
#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  kernel_id Kernel_id =
      ext::oneapi::experimental::get_kernel_id<(void (*)(int *, int))ff_1>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, value);
    Handler.parallel_for(R2, Kernel);
  });
  Queue.wait();
  PassB = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 2b: " << (PassB ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
#endif
  return PassA && PassB;
}

// Templated free function definition.
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    ext::oneapi::experimental::nd_range_kernel<2>)) void ff_3(T *ptr, T start) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::this_work_item::get_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + start;
}

// Explicit instantiation with “int*”.
template void ff_3(int *ptr, int start);

bool test_3(queue Queue) {
  constexpr int Range = 16;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int value = 55;
  int Result[Range] = {55, 56, 55, 56, 56, 57, 56, 57,
                       55, 56, 55, 56, 56, 57, 56, 57};
  nd_range<2> R2{range<2>{4, 4}, range<2>{2, 2}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R2, [=](nd_item<2> Item) {
      int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(usmPtr);
      id<2> GId = Item.get_global_id();
      id<2> LId = Item.get_local_id();
      ptr2D[GId.get(0)][GId.get(1)] = LId.get(0) + LId.get(1) + value;
    });
  });
  Queue.wait();
  bool PassA = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 3a: " << (PassA ? "PASS" : "FAIL") << std::endl;

  bool PassB = false;
#ifndef __SYCL_DEVICE_ONLY__
  kernel_bundle Bundle =
      get_kernel_bundle<bundle_state::executable>(Queue.get_context());
  kernel_id Kernel_id = ext::oneapi::experimental::get_kernel_id<(
      void (*)(int *, int))ff_3<int>>();
  kernel Kernel = Bundle.get_kernel(Kernel_id);
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, value);
    Handler.parallel_for(R2, Kernel);
  });
  Queue.wait();
  PassB = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 3b: " << (PassB ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
#endif
  return PassA && PassB;
}

int main() {
  queue Queue;

  bool Pass = true;
  Pass &= test_0(Queue);
  Pass &= test_1(Queue);
  Pass &= test_2(Queue);
  Pass &= test_3(Queue);

  return Pass ? 0 : 1;
}
