// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test tests free function kernel code generation and execution.

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

// Kernel finder
class KernelFinder {
  queue &Queue;
  std::vector<sycl::kernel_id> AllKernelIDs;

public:
  KernelFinder(queue &Q) : Queue(Q) {
    // Obtain kernel bundle
    kernel_bundle Bundle =
        get_kernel_bundle<bundle_state::executable>(Queue.get_context());
    std::cout << "Bundle obtained\n";
    AllKernelIDs = sycl::get_kernel_ids();
    std::cout << "Number of kernels = " << AllKernelIDs.size() << std::endl;
    for (auto K : AllKernelIDs) {
      std::cout << "Kernel obtained: " << K.get_name() << std::endl;
    }
    std::cout << std::endl;
  }

  kernel get_kernel(const char *name) {
    kernel_bundle Bundle =
        get_kernel_bundle<bundle_state::executable>(Queue.get_context());
    for (auto K : AllKernelIDs) {
      auto Kname = K.get_name();
      if (strcmp(name, Kname) == 0) {
        kernel Kernel = Bundle.get_kernel(K);
        return Kernel;
      }
    }
    std::cout << "No kernel named " << name << " found\n";
    exit(1);
  }
};

struct Simple {
  int x;
  char c[100];
  float f;
};

struct WithPointer {
  int x;
  float *fp;
  float f;
};

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

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void ff_0(int *ptr, int start, int end) {
  for (int i = start; i <= end; i++)
    ptr[i] = start + end;
}

bool test_0(queue Queue, KernelFinder &KF) {
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
  bool Passa = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 0a: " << (Passa ? "PASS" : "FAIL") << std::endl;

  kernel Kernel = KF.get_kernel("__free_function_ff_0");
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, start);
    Handler.set_arg(2, end);
    Handler.single_task(Kernel);
  });
  Queue.wait();
  bool Passb = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 0b: " << (Passb ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
  return Passa && Passb;
}

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((ext::oneapi::experimental::nd_range_kernel<1>))
void ff_1(int *ptr, int start, int end) {
  nd_item<1> Item = ext::oneapi::experimental::this_nd_item<1>();
  id<1> GId = Item.get_global_id();
  ptr[GId.get(0)] = GId.get(0) + start + end;
}

bool test_1(queue Queue, KernelFinder &KF) {
  constexpr int Range = 10;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int start = 3;
  struct Simple S {
    66, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 7.1
  };
  int Result[Range] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
  nd_range<1> R1{ {Range},{1} };

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R1, [=](nd_item<1> Item) {
      id<1> GId = Item.get_global_id();
      usmPtr[GId.get(0)] = GId.get(0) + start + Range;
    });
  });
  Queue.wait();
  bool Passa = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 1a: " << (Passa ? "PASS" : "FAIL") << std::endl;

  kernel Kernel = KF.get_kernel("__free_function_ff_1");
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, start);
    Handler.set_arg(2, Range);
    Handler.parallel_for(R1, Kernel);
  });
  Queue.wait();
  bool Passb = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 1b: " << (Passb ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
  return Passa && Passb;
}

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<2>))
void ff_2(int *ptr, int start, struct Simple S) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::experimental::this_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] =
      LId.get(0) + LId.get(1) + start + S.x + S.f + S.c[2];
}

bool test_2(queue Queue, KernelFinder &KF) {
  constexpr int Range = 16;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int value = 55;
  struct Simple S {
    66, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 7.1
  };
  int Result[Range] = {130, 131, 130, 131, 131, 132, 131, 132,
                       130, 131, 130, 131, 131, 132, 131, 132};
  nd_range<2> R2{range<2>{4, 4}, range<2>{2, 2}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R2, [=](nd_item<2> Item) {
      int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(usmPtr);
      id<2> GId = Item.get_global_id();
      id<2> LId = Item.get_local_id();
      ptr2D[GId.get(0)][GId.get(1)] =
          LId.get(0) + LId.get(1) + value + S.x + S.f + S.c[2];
    });
  });
  Queue.wait();
  bool Passa = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 2a: " << (Passa ? "PASS" : "FAIL") << std::endl;

  kernel Kernel = KF.get_kernel("__free_function_ff_2");
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, value);
    Handler.set_arg(2, S);
    Handler.parallel_for(R2, Kernel);
  });
  Queue.wait();
  bool Passb = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 2b: " << (Passb ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
  return Passa && Passb;
}

// Templated free function definition
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    ext::oneapi::experimental::nd_range_kernel<2>)) void ff_3(T *ptr, T start,
                                                              struct Simple S) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::experimental::this_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] =
      LId.get(0) + LId.get(1) + start + S.x + S.f + S.c[2];
}

// Explicit instantiation with “int*”
template void ff_3(int *ptr, int start, struct Simple S);

bool test_3(queue Queue, KernelFinder &KF) {
  constexpr int Range = 16;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  int value = 55;
  struct Simple S {
    66, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 7.1
  };
  int Result[Range] = {130, 131, 130, 131, 131, 132, 131, 132,
                       130, 131, 130, 131, 131, 132, 131, 132};
  nd_range<2> R2{range<2>{4, 4}, range<2>{2, 2}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R2, [=](nd_item<2> Item) {
      int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(usmPtr);
      id<2> GId = Item.get_global_id();
      id<2> LId = Item.get_local_id();
      ptr2D[GId.get(0)][GId.get(1)] =
          LId.get(0) + LId.get(1) + value + S.x + S.f + S.c[2];
    });
  });
  Queue.wait();
  bool Passa = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 3a: " << (Passa ? "PASS" : "FAIL") << std::endl;

  kernel Kernel = KF.get_kernel("_ZTSFvPii6SimpleE");
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, value);
    Handler.set_arg(2, S);
    Handler.parallel_for(R2, Kernel);
  });
  Queue.wait();
  bool Passb = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 3b: " << (Passb ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
  return Passa && Passb;
}

// Templated free function definition
template <typename T>
SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((
    ext::oneapi::experimental::nd_range_kernel<2>)) void ff_4(T *ptr, T start,
                                                              struct WithPointer
                                                                  S) {
  int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(ptr);
  nd_item<2> Item = ext::oneapi::experimental::this_nd_item<2>();
  id<2> GId = Item.get_global_id();
  id<2> LId = Item.get_local_id();
  ptr2D[GId.get(0)][GId.get(1)] =
      LId.get(0) + LId.get(1) + start + S.x + S.f + *S.fp;
}

// Explicit instantiation with “int*”
template void ff_4(int *ptr, int start, struct WithPointer S);

bool test_4(queue Queue, KernelFinder &KF) {
  constexpr int Range = 16;
  int *usmPtr = malloc_shared<int>(Range, Queue);
  float *fp = malloc_shared<float>(1, Queue);
  *fp = 8.2;
  int value = 55;
  struct WithPointer S {
    3, fp, 7.1
  };
  int Result[Range] = {73, 74, 73, 74, 74, 75, 74, 75,
                       73, 74, 73, 74, 74, 75, 74, 75};
  nd_range<2> R2{range<2>{4, 4}, range<2>{2, 2}};

  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.parallel_for(R2, [=](nd_item<2> Item) {
      int(&ptr2D)[4][4] = *reinterpret_cast<int(*)[4][4]>(usmPtr);
      id<2> GId = Item.get_global_id();
      id<2> LId = Item.get_local_id();
      ptr2D[GId.get(0)][GId.get(1)] =
          LId.get(0) + LId.get(1) + value + S.x + S.f + *S.fp;
    });
  });
  Queue.wait();
  bool Passa = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 4a: " << (Passa ? "PASS" : "FAIL") << std::endl;

  kernel Kernel = KF.get_kernel("_ZTSFvPii11WithPointerE");
  memset(usmPtr, 0, Range * sizeof(int));
  Queue.submit([&](handler &Handler) {
    Handler.set_arg(0, usmPtr);
    Handler.set_arg(1, value);
    Handler.set_arg(2, S);
    Handler.parallel_for(R2, Kernel);
  });
  Queue.wait();
  bool Passb = checkUSM(usmPtr, Range, Result);
  std::cout << "Test 4b: " << (Passb ? "PASS" : "FAIL") << std::endl;

  free(usmPtr, Queue);
  return Passa && Passb;
}

int main() {
  queue Queue;
  KernelFinder KF(Queue);

  bool Pass = true;
  Pass &= test_0(Queue, KF);
  Pass &= test_1(Queue, KF);
  Pass &= test_2(Queue, KF);
  Pass &= test_3(Queue, KF);
  Pass &= test_4(Queue, KF);

  return Pass ? 0 : 1;
}
