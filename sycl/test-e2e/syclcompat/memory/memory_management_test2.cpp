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
 *  memory_management_test2.cpp
 *
 *  Description:
 *    memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ memory_management_test2.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <syclcompat/memory.hpp>

#include "memory_common.hpp"

constexpr size_t DataW = 100;
constexpr size_t DataH = 100;

void test_memcpy_pitched() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  size_t width = 6;
  size_t height = 8;
  float *h_data;
  float *h_ref;
  size_t h_pitch = sizeof(float) * width;
  h_data = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_data[i] = (float)i;

  h_ref = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_ref[i] = (float)i;

  // alloc device memory.
  size_t d_pitch;
  float *d_data;
  d_data = (float *)syclcompat::malloc(d_pitch, sizeof(float) * width, height);

  // copy to Device.
  syclcompat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width,
                     height);

  // copy back to host.
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                     height);

  check(h_data, h_ref, width * height);

  // memset device data.
  syclcompat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                     height);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width * height);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data);
}

void test_memcpy_kernel() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int Offset =
      0; // Current dpcpp version in ics environment has bugs with Offset >
         // 0, CORC-6222 has fixed this issue, but the version of dpcpp used in
         // ics environment has not cover this patch. After it has this patch,
         // Offest could be set to 100, and current test case will pass.

  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  // syclcompat::dev_mgr::instance().select_device(0);

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  d_A = (float *)syclcompat::malloc(Num * sizeof(float));
  d_B = (float *)syclcompat::malloc(Num * sizeof(float));
  d_C = (float *)syclcompat::malloc(Num * sizeof(float));
  syclcompat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(float));
  syclcompat::memcpy((void *)d_B, (void *)h_B, Num * sizeof(float));

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    syclcompat::get_default_queue().submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>(Num - Offset), [=](sycl::id<1> id) {
        float *A = d_A;
        float *B = d_B;
        float *C = d_C;
        int i = id[0];
        C[i] = A[i] + B[i];
      });
    });
    syclcompat::get_default_queue().wait_and_throw();
  }

  syclcompat::memcpy((void *)(h_C + Offset), (void *)d_C,
                     (Num - Offset) * sizeof(float));
  syclcompat::free((void *)d_A);
  syclcompat::free((void *)d_B);
  syclcompat::free((void *)d_C);

  // verify
  for (int i = Offset; i < Num; i++) {
    assert(fabs(h_C[i] - h_A[i] - h_B[i]) <= 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_global_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  float h_A[DataW][DataH];
  float h_B[DataW][DataH];
  float h_C[DataW][DataH];

  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 2.0f;
    }
  }

  syclcompat::global_memory<float, 2> g_A(DataW, DataH);
  syclcompat::global_memory<float, 2> g_B(DataW, DataH);
  syclcompat::global_memory<float, 2> g_C(DataW, DataH);

  g_A.init();
  g_B.init();
  g_C.init();

  syclcompat::memcpy((void *)g_A.get_ptr(), (void *)&h_A[0][0],
                     DataW * DataH * sizeof(float));
  syclcompat::memcpy((void *)g_B.get_ptr(), (void *)&h_B[0][0],
                     DataW * DataH * sizeof(float));

  {
    syclcompat::get_default_queue().submit([&](sycl::handler &cgh) {
      auto g_A_acc = g_A.get_access(cgh);
      auto g_B_acc = g_B.get_access(cgh);
      auto g_C_acc = g_C.get_access(cgh);
      cgh.parallel_for(sycl::range<2>(DataW, DataH), [=](sycl::id<2> id) {
        // test_feature:accessor
        // test_feature:memory_region
        syclcompat::accessor<float, syclcompat::memory_region::global, 2> A(
            g_A_acc);
        // test_feature:accessor
        // test_feature:memory_region
        syclcompat::accessor<float, syclcompat::memory_region::global, 2> B(
            g_B_acc);
        // test_feature:accessor
        // test_feature:memory_region
        syclcompat::accessor<float, syclcompat::memory_region::global, 2> C(
            g_C_acc);
        int i = id[0], j = id[1];
        C[i][j] = A[i][j] + B[i][j];
      });
    });
    syclcompat::get_default_queue().wait_and_throw();
  }
  syclcompat::memcpy((void *)&h_C[0][0], (void *)g_C.get_ptr(),
                     DataW * DataH * sizeof(float));

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      assert(fabs(h_C[i][j] - h_A[i][j] - h_B[i][j]) <= 1e-5);
    }
  }
}

void test_constant_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  float h_A[DataW][DataH];
  float h_B[DataW][DataH];
  float h_C[DataW][DataH];

  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 2.0f;
    }
  }

  syclcompat::constant_memory<float, 2> c_A(DataW, DataH);
  syclcompat::constant_memory<float, 2> c_B(DataW, DataH);
  syclcompat::global_memory<float, 2> g_C(DataW, DataH);

  c_A.init();
  c_B.init();
  g_C.init();
  syclcompat::memcpy((void *)c_A.get_ptr(), (void *)&h_A[0][0],
                     DataW * DataH * sizeof(float));
  syclcompat::memcpy((void *)c_B.get_ptr(), (void *)&h_B[0][0],
                     DataW * DataH * sizeof(float));

  {
    syclcompat::get_default_queue().submit([&](sycl::handler &cgh) {
      auto c_A_acc = c_A.get_access(cgh);
      auto c_B_acc = c_B.get_access(cgh);
      auto g_C_acc = g_C.get_access(cgh);
      cgh.parallel_for(sycl::range<2>(DataW, DataH), [=](sycl::id<2> id) {
        syclcompat::accessor<float, syclcompat::memory_region::constant, 2> A(
            c_A_acc);
        syclcompat::accessor<float, syclcompat::memory_region::constant, 2> B(
            c_B_acc);
        syclcompat::accessor<float, syclcompat::memory_region::global, 2> C(
            g_C_acc);
        int i = id[0], j = id[1];
        C[i][j] = A[i][j] + B[i][j];
      });
    });
    syclcompat::get_default_queue().wait_and_throw();
  }
  syclcompat::memcpy((void *)&h_C[0][0], (void *)g_C.get_ptr(),
                     DataW * DataH * sizeof(float));
  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      assert(fabs(h_C[i][j] - h_A[i][j] - h_B[i][j]) <= 1e-5);
    }
  }
}

void test_memcpy_pitched_q() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  sycl::queue q{{sycl::property::queue::in_order()}};
  size_t width = 6;
  size_t height = 8;
  float *h_data;
  float *h_ref;
  size_t h_pitch = sizeof(float) * width;
  h_data = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_data[i] = (float)i;

  h_ref = (float *)malloc(sizeof(float) * width * height);
  for (int i = 0; i < width * height; i++)
    h_ref[i] = (float)i;

  // alloc device memory.
  size_t d_pitch;
  float *d_data;
  d_data =
      (float *)syclcompat::malloc(d_pitch, sizeof(float) * width, height, q);

  // copy to Device.
  syclcompat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width,
                     height, q);

  // copy back to host.
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                     height, q);

  check(h_data, h_ref, width * height);

  // memset device data.
  syclcompat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height, q);

  // copy back to host
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                     height, q);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width * height);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data, q);
}

int main() {
  test_memcpy_kernel();
  test_memcpy_pitched();
  test_memcpy_pitched_q();

  test_global_memory();
  test_constant_memory();
  return 0;
}
