/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat API
 *
 *  kernel_function_pointer.cpp
 *
 *  Description:
 *    kernel function pointer header API tests
 **************************************************************************/
// The original source was under the license below:
// ====------ kernel_function_pointer.cpp---------- -*- C++ -*
// ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===---------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>

#include <sycl/sycl.hpp>

#include <syclcompat/defs.hpp>
#include <syclcompat/device.hpp>
#include <syclcompat/kernel.hpp>
#include <syclcompat/memory.hpp>

void vectorAdd(const int *A, int *B, int *C, int N,
               const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// SYCL kernel wrapper.
void vectorAdd_wrapper(const int *A, int *B, int *C, int N) {
  sycl::queue queue = *syclcompat::kernel_launcher::_que;
  unsigned int localMemSize = syclcompat::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = syclcompat::kernel_launcher::_nr;

  queue.parallel_for(
      nr, [=](sycl::nd_item<3> item_ct1) { vectorAdd(A, B, C, N, item_ct1); });
}

template <typename T>
void vectorTemplateAdd(const T *A, T *B, T *C, int N,
                       const sycl::nd_item<3> &item_ct1) {
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// SYCL kernel wrapper.
template <typename T>
void vectorTemplateAdd_wrapper(const T *A, T *B, T *C, int N) {
  sycl::queue queue = *syclcompat::kernel_launcher::_que;
  unsigned int localMemSize = syclcompat::kernel_launcher::_local_mem_size;
  sycl::nd_range<3> nr = syclcompat::kernel_launcher::_nr;

  queue.parallel_for(nr, [=](sycl::nd_item<3> item_ct1) {
    vectorTemplateAdd<T>(A, B, C, N, item_ct1);
  });
}

void hostCallback(void *userData) {
  const char *msg = static_cast<const char *>(userData);
  std::cout << "Host callback executed. Message: " << msg << std::endl;
}

template <typename T> using fpt = void (*)(const T *, T *, T *, int);

void test_kernel_launch() {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int N = 10;
  size_t size = N * sizeof(int);

  int *h_A = new int[N];
  int *h_B = new int[N];
  int *h_C = new int[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<int>(i);
    h_B[i] = static_cast<int>(i * 2);
  }

  int *d_A, *d_B, *d_C;
  d_A = (int *)sycl::malloc_device(size, q_ct1);
  d_B = (int *)sycl::malloc_device(size, q_ct1);
  d_C = (int *)sycl::malloc_device(size, q_ct1);

  q_ct1.memcpy(d_A, h_A, size).wait();
  q_ct1.memcpy(d_B, h_B, size).wait();

  fpt<int> fp = syclcompat::wrapper_register(vectorAdd_wrapper).get();

  void *kernel_func =
      (void *)syclcompat::wrapper_register(&vectorAdd_wrapper).get();

  syclcompat::kernel_launcher::launch(fp, 1, 10, 0, 0, d_A, d_B, d_C, N);

  q_ct1.memcpy(h_C, d_C, size).wait();

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }

  void *args[4];
  args[0] = &d_A;
  args[1] = &d_B;
  args[2] = &d_C;
  args[3] = &N;

  syclcompat::kernel_launcher::launch(fp, 1, 10, args, 0, 0);

  q_ct1.memcpy(h_C, d_C, size).wait();

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }

  syclcompat::kernel_launcher::launch(fp, 1, 10, args, 0, 0);

  q_ct1.memcpy(h_C, d_C, size).wait();

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }

  syclcompat::wait_and_free(d_A, q_ct1);
  syclcompat::wait_and_free(d_B, q_ct1);
  syclcompat::wait_and_free(d_C, q_ct1);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

template <typename T> void goo(fpt<T> p) {
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  int N = 10;
  size_t size = N * sizeof(int);

  int *h_A = new int[N];
  int *h_B = new int[N];
  int *h_C = new int[N];

  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<int>(i);
    h_B[i] = static_cast<int>(i * 2);
  }

  int *d_A, *d_B, *d_C;
  d_A = (int *)sycl::malloc_device(size, q_ct1);
  d_B = (int *)sycl::malloc_device(size, q_ct1);
  d_C = (int *)sycl::malloc_device(size, q_ct1);

  q_ct1.memcpy(d_A, h_A, size).wait();
  q_ct1.memcpy(d_B, h_B, size).wait();

  syclcompat::kernel_launcher::launch(p, 1, 10, 0, 0, d_A, d_B, d_C, N);

  q_ct1.memcpy(h_C, d_C, size).wait();

  std::cout << "Result: " << std::endl;
  for (int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      std::cout << "test failed" << std::endl;
      exit(-1);
    }
    std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
  }

  syclcompat::wait_and_free(d_A, q_ct1);
  syclcompat::wait_and_free(d_B, q_ct1);
  syclcompat::wait_and_free(d_C, q_ct1);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

template <typename T> void test_wrapper_register() {
  fpt<int> a =
      syclcompat::wrapper_register<decltype(a)>(vectorTemplateAdd_wrapper)
          .get();
  goo<T>(syclcompat::wrapper_register<
             typename syclcompat::nth_argument_type<decltype(goo<T>), 0>::type>(
             vectorTemplateAdd_wrapper)
             .get());
}

void test_host_callback() {
  syclcompat::host_func fn = hostCallback;
  sycl::queue q_ct1 = syclcompat::get_default_queue();
  const char *message = "Execution finished.";
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() { fn((void *)message); });
  };
}

int main() {
  test_kernel_launch();
  test_wrapper_register<int>();
  test_host_callback();
  std::cout << "test success" << std::endl;
  return 0;
}
