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
 *  SYCL compatibility API
 *
 *  MemoryManagementTest2.cpp
 *
 *  Description:
 *    memory operations tests
 **************************************************************************/

// The original source was under the license below:
// ====------ MemoryManagementTest2.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/compat.hpp>
#include <sycl/sycl.hpp>

#include "memory_check.hpp"

using namespace sycl::ext::oneapi::experimental;

TEST(Memory, memcpy_pitched) {
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
  d_data = (float *)compat::malloc(d_pitch, sizeof(float) * width, height);

  // copy to Device.
  compat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width,
                 height);

  // copy back to host.
  compat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                 height);

  check(h_data, h_ref, width, height, 1);

  // memset device data.
  compat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  compat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                 height);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  compat::free((void *)d_data);
}

TEST(Memory, memcpy_kernel) {

  int Num = 5000;
  int Offset =
      0; // Current dpcpp version in ics environment has bugs with Offset >
         // 0, CORC-6222 has fixed this issue, but the version of dpcpp used in
         // ics environment has not cover this patch. After it has this patch,
         // Offest could be set to 100, and current test case will pass.

  float *h_A = (float *)malloc(Num * sizeof(float));
  float *h_B = (float *)malloc(Num * sizeof(float));
  float *h_C = (float *)malloc(Num * sizeof(float));

  // compat::dev_mgr::instance().select_device(0);

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  d_A = (float *)compat::malloc(Num * sizeof(float));
  d_B = (float *)compat::malloc(Num * sizeof(float));
  d_C = (float *)compat::malloc(Num * sizeof(float));
  compat::memcpy((void *)d_A, (void *)h_A, Num * sizeof(float));
  compat::memcpy((void *)d_B, (void *)h_B, Num * sizeof(float));

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    compat::get_default_queue().submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>(Num - Offset), [=](sycl::id<1> id) {
        float *A = d_A;
        float *B = d_B;
        float *C = d_C;
        int i = id[0];
        C[i] = A[i] + B[i];
      });
    });
    compat::get_default_queue().wait_and_throw();
  }

  compat::memcpy((void *)(h_C + Offset), (void *)d_C,
                 (Num - Offset) * sizeof(float));
  compat::free((void *)d_A);
  compat::free((void *)d_B);
  compat::free((void *)d_C);

  // verify
  for (int i = Offset; i < Num; i++) {
    EXPECT_LE(fabs(h_C[i] - h_A[i] - h_B[i]), 1e-5);
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

#define DataW 100
#define DataH 100

compat::global_memory<float, 2> g_A(DataW, DataH);
compat::global_memory<float, 2> g_B(DataW, DataH);
compat::global_memory<float, 2> g_C(DataW, DataH);

TEST(Memory, global_memory) {

  float h_A[DataW][DataH];
  float h_B[DataW][DataH];
  float h_C[DataW][DataH];

  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      h_A[i][j] = 1.0f;
      h_B[i][j] = 2.0f;
    }
  }

  g_A.init();
  g_B.init();
  g_C.init();

  compat::memcpy((void *)g_A.get_ptr(), (void *)&h_A[0][0],
                 DataW * DataH * sizeof(float));
  compat::memcpy((void *)g_B.get_ptr(), (void *)&h_B[0][0],
                 DataW * DataH * sizeof(float));

  {
    compat::get_default_queue().submit([&](sycl::handler &cgh) {
      auto g_A_acc = g_A.get_access(cgh);
      auto g_B_acc = g_B.get_access(cgh);
      auto g_C_acc = g_C.get_access(cgh);
      cgh.parallel_for(sycl::range<2>(DataW, DataH), [=](sycl::id<2> id) {
        // test_feature:accessor
        // test_feature:memory_region
        compat::accessor<float, compat::memory_region::global, 2> A(g_A_acc);
        // test_feature:accessor
        // test_feature:memory_region
        compat::accessor<float, compat::memory_region::global, 2> B(g_B_acc);
        // test_feature:accessor
        // test_feature:memory_region
        compat::accessor<float, compat::memory_region::global, 2> C(g_C_acc);
        int i = id[0], j = id[1];
        C[i][j] = A[i][j] + B[i][j];
      });
    });
    compat::get_default_queue().wait_and_throw();
  }
  compat::memcpy((void *)&h_C[0][0], (void *)g_C.get_ptr(),
                 DataW * DataH * sizeof(float));

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      EXPECT_LE(fabs(h_C[i][j] - h_A[i][j] - h_B[i][j]), 1e-5);
    }
  }
}

compat::shared_memory<float, 1> s_A(DataW);
compat::shared_memory<float, 1> s_B(DataW);
compat::shared_memory<float, 1> s_C(DataW);

TEST(Memory, shared_memory) {

  s_A.init();
  s_B.init();
  s_C.init();

  for (int i = 0; i < DataW; i++) {
    s_A[i] = 1.0f;
    s_B[i] = 2.0f;
  }

  {
    compat::get_default_queue().submit([&](sycl::handler &cgh) {
      float *d_A = s_A.get_ptr();
      float *d_B = s_B.get_ptr();
      float *d_C = s_C.get_ptr();
      cgh.parallel_for(sycl::range<1>(DataW), [=](sycl::id<1> id) {
        int i = id[0];
        float *A = d_A;
        float *B = d_B;
        float *C = d_C;
        C[i] = A[i] + B[i];
      });
    });
    compat::get_default_queue().wait_and_throw();
  }

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      EXPECT_LE(fabs(s_C[i] - s_A[i] - s_B[i]), 1e-5);
    }
  }
}

TEST(Memory, memcpy_pitched_q) {
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
  d_data = (float *)compat::malloc(d_pitch, sizeof(float) * width, height, q);

  // copy to Device.
  compat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width,
                 height, q);

  // copy back to host.
  compat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                 height, q);

  check(h_data, h_ref, width, height, 1);

  // memset device data.
  compat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height, q);

  // copy back to host
  compat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width,
                 height, q);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  compat::free((void *)d_data, q);
}
