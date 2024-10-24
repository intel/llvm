// ====------ memory_management_test_mempcy_2_usmnone.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define SYCLCOMPAT_USM_LEVEL_NONE
#include <sycl/detail/core.hpp>
#include <syclcompat/memory.hpp>
#include "memory_common.hpp"

void check(float *h_data, float *h_ref, size_t width, size_t height,
           size_t depth) {
  for (int i = 0; i < width * height * depth; i++) {
    float diff = fabs(h_data[i] - h_ref[i]);
    if (diff > 1.e-6) {
      printf("Verification failed!");
      printf("h_data[%d]=%f, h_ref[%d]=%f, diff=%f\n", i, h_data[i], i,
             h_ref[i], diff);
      exit(-1);
    }
  }
}


void test_mempcy_pitched() {
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
  syclcompat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height);

  // copy back to host.
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height);

  check(h_data, h_ref, width, height, 1);

  // memset device data.
  syclcompat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data);
}

void test_memcpy_reinterp_kernel() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

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
  syclcompat::memcpy((void*) d_A, (void*) h_A, Num * sizeof(float));
  syclcompat::memcpy((void*) d_B, (void*) h_B, Num * sizeof(float));

  {
    syclcompat::buffer_t buffer_A = syclcompat::get_buffer(d_A);
    syclcompat::buffer_t buffer_B = syclcompat::get_buffer(d_B);
    syclcompat::buffer_t buffer_C = syclcompat::get_buffer(d_C);

    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto A = buffer_A.reinterpret<float>().get_access<sycl::access::mode::read_write>(cgh);
      auto B = buffer_B.reinterpret<float>().get_access<sycl::access::mode::read_write>(cgh);
      auto C = buffer_C.reinterpret<float>().get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
          sycl::range<1>(Num),
          [=](sycl::id<1> id) {
             int i = id[0];

            C[i] = A[i] + B[i];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }

  syclcompat::memcpy((void*) (h_C), (void*) d_C, (Num) * sizeof(float));
  syclcompat::free((void*)d_A);
  syclcompat::free((void*)d_B);
  syclcompat::free((void*)d_C);

  // verify
  for(int i = 0; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f, C = %f:\n", i,h_A[i],  h_B[i],  h_C[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_memcpy_kernel() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

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
  syclcompat::memcpy((void*) d_A, (void*) h_A, Num * sizeof(float));
  syclcompat::memcpy((void*) d_B, (void*) h_B, Num * sizeof(float));

  {
    auto buffer_A = syclcompat::get_buffer<float>(d_A);
    auto buffer_B = syclcompat::get_buffer<float>(d_B);
    auto buffer_C = syclcompat::get_buffer<float>(d_C);

    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto A = buffer_A.get_access<sycl::access::mode::read_write>(cgh);
      auto B = buffer_B.get_access<sycl::access::mode::read_write>(cgh);
      auto C = buffer_C.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
          sycl::range<1>(Num),
          [=](sycl::id<1> id) {
             int i = id[0];

            C[i] = A[i] + B[i];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }

  syclcompat::memcpy((void*) (h_C), (void*) d_C, Num * sizeof(float));
  syclcompat::free((void*)d_A);
  syclcompat::free((void*)d_B);
  syclcompat::free((void*)d_C);

  // verify
  for(int i = 0; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f, C = %f:\n", i,h_A[i],  h_B[i],  h_C[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_access_wrapper() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int Offset = 0; // Current dpcpp version in ics environment has bugs with Offset > 0,
                  // CORC-6222 has fixed this issue, but the version of dpcpp used in ics
                  // environment has not cover this patch. After it has this patch,
                  // Offest could be set to 100, and current test case will pass.

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

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
  syclcompat::memcpy((void*) d_A, (void*) h_A, Num * sizeof(float));
  syclcompat::memcpy((void*) d_B, (void*) h_B, Num * sizeof(float));

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      syclcompat::access_wrapper<float *> d_A_acc(d_A, cgh);
      syclcompat::access_wrapper<float *> d_B_acc(d_B, cgh);
      syclcompat::access_wrapper<float *> d_C_acc(d_C, cgh);

        cgh.parallel_for(
          sycl::range<1>(Num-Offset),
          [=](sycl::id<1> id) {

            float *A = d_A_acc.get_raw_pointer();
            float *B = d_B_acc.get_raw_pointer();
            float *C = d_C_acc.get_raw_pointer();
             int i = id[0];
            C[i] = A[i] + B[i];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }

  syclcompat::memcpy((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float));
  syclcompat::free((void*)d_A);
  syclcompat::free((void*)d_B);
  syclcompat::free((void*)d_C);

  // verify
  for(int i = Offset; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f, C = %f:\n", i,h_A[i],  h_B[i],  h_C[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

#define DataW 100
#define DataH 100
syclcompat::constant_memory<float, 2> c_A(DataW, DataH);
syclcompat::constant_memory<float, 2> c_B(DataW, DataH);
syclcompat::global_memory<float, 2> c_C(DataW, DataH);

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

  c_A.init();
  c_B.init();
  c_C.init();
  syclcompat::memcpy((void *)c_A.get_ptr(), (void *)&h_A[0][0], DataW * DataH * sizeof(float));
  syclcompat::memcpy((void *)c_B.get_ptr(), (void *)&h_B[0][0], DataW * DataH * sizeof(float));

  {
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto c_A_acc = c_A.get_access(cgh);
      auto c_B_acc = c_B.get_access(cgh);
      auto c_C_acc = c_C.get_access(cgh);
        cgh.parallel_for(
          sycl::range<2>(DataW, DataH),
          [=](sycl::id<2> id) {
            syclcompat::accessor<float, syclcompat::memory_region::constant, 2> A(c_A_acc);
            syclcompat::accessor<float, syclcompat::memory_region::constant, 2> B(c_B_acc);
            syclcompat::accessor<float, syclcompat::memory_region::global, 2> C(c_C_acc);
            int i = id[0], j = id[1];
            C[i][j] = A[i][j] + B[i][j];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }
  syclcompat::memcpy((void *)&h_C[0][0], (void *)c_C.get_ptr(), DataW * DataH * sizeof(float));

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      if (fabs(h_C[i][j] - h_A[i][j] - h_B[i][j]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element [%d][%d]:\n", i, j);
        exit(EXIT_FAILURE);
      }
    }
  }
}

syclcompat::global_memory<float, 2> g_A(DataW, DataH);
syclcompat::global_memory<float, 2> g_B(DataW, DataH);
syclcompat::global_memory<float, 2> g_C(DataW, DataH);

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

  g_A.init();
  g_B.init();
  g_C.init();

  syclcompat::memcpy((void *)g_A.get_ptr(), (void *)&h_A[0][0], DataW * DataH * sizeof(float));
  syclcompat::memcpy((void *)g_B.get_ptr(), (void *)&h_B[0][0], DataW * DataH * sizeof(float));

  {
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto g_A_acc = g_A.get_access(cgh);
      auto g_B_acc = g_B.get_access(cgh);
      auto g_C_acc = g_C.get_access(cgh);
        cgh.parallel_for(
          sycl::range<2>(DataW, DataH),
          [=](sycl::id<2> id) {
            syclcompat::accessor<float, syclcompat::memory_region::global, 2> A(g_A_acc);
            syclcompat::accessor<float, syclcompat::memory_region::global, 2> B(g_B_acc);
            syclcompat::accessor<float, syclcompat::memory_region::global, 2> C(g_C_acc);
            int i = id[0], j = id[1];
            C[i][j] = A[i][j] + B[i][j];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }
  syclcompat::memcpy((void *)&h_C[0][0], (void *)g_C.get_ptr(), DataW * DataH * sizeof(float));

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      if (fabs(h_C[i][j] - h_A[i][j] - h_B[i][j]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element [%d][%d]:\n", i, j);
        exit(EXIT_FAILURE);
      }
    }
  }
}

syclcompat::shared_memory<float, 1> s_A(DataW);
syclcompat::shared_memory<float, 1> s_B(DataW);
syclcompat::shared_memory<float, 1> s_C(DataW);

void test_shared_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  s_A.init();
  s_B.init();
  s_C.init();

  for (int i = 0; i < DataW; i++) {
    s_A[i] = 1.0f;
    s_B[i] = 2.0f;
  }

  {
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
        syclcompat::access_wrapper<float *> A_acc(s_A.get_ptr(), cgh);
        syclcompat::access_wrapper<float *> B_acc(s_B.get_ptr(), cgh);
        syclcompat::access_wrapper<float *> C_acc(s_C.get_ptr(), cgh);
        cgh.parallel_for(
          sycl::range<1>(DataW),
          [=](sycl::id<1> id) {
            int i = id[0];
            float * A = A_acc.get_raw_pointer();
            float * B = B_acc.get_raw_pointer();
            float * C = C_acc.get_raw_pointer();
            C[i] = A[i] + B[i];
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }

  // verify hostD
  for (int i = 0; i < DataW; i++) {
    for (int j = 0; j < DataH; j++) {
      if (fabs(s_C[i] - s_A[i] - s_B[i]) > 1e-5) {
        fprintf(stderr, "Result verification failed at element [%d][%d]:\n", i, j);
        exit(EXIT_FAILURE);
      }
    }
  }
}

void test_local_memory() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 16;

  float *h_A = (float*)malloc(Num * Num * sizeof(float));
  float *h_B = (float*)malloc(Num * Num * sizeof(float));

  for (int i = 0; i < Num; i++) {
    for(int j = 0; j < Num; j++) {
      h_A[i * Num + j] = 2.0f;
    }
  }

  float *d_A;
  d_A = (float *)syclcompat::malloc(Num * Num * sizeof(float));

  {
    auto buffer_A = syclcompat::get_buffer<float>(d_A);

    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      sycl::range<2> acc_range(Num, Num);
      sycl::local_accessor<float, 2> C_local_acc(acc_range, cgh);
      auto A = buffer_A.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(Num, Num), sycl::range<2>(Num, Num)),
          [=](sycl::nd_item<2> id) {
            syclcompat::accessor<float, syclcompat::memory_region::local, 2> C_local(C_local_acc, acc_range);
            int i = id.get_local_id(0), j = id.get_local_id(1);
            C_local[i][j] = 1;
            A[i * Num + j] = C_local[i][j] * 2;
          });
      });
      syclcompat::get_default_queue().wait_and_throw();
  }

  syclcompat::memcpy((void*) (h_B), (void*) d_A, Num * Num * sizeof(float));
  syclcompat::free((void*)d_A);

  // verify
  for(int i = 0; i < Num * Num; i++){
      if (fabs(h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f\n", i, h_A[i],  h_B[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
}

void test_mempcy_pitched(sycl::queue &q) {
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
  d_data = (float *)syclcompat::malloc(d_pitch, sizeof(float) * width, height, q);

  // copy to Device.
  syclcompat::memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height, q);

  // copy back to host.
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, q);

  check(h_data, h_ref, width, height, 1);

  // memset device data.
  syclcompat::memset(d_data, d_pitch, 0x1, sizeof(float) * width, height, q);

  // copy back to host
  syclcompat::memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, q);

  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data, q);

  printf("test_mempcy_pitched passed!\n");
}

int main() {
  test_mempcy_pitched();
  test_memcpy_reinterp_kernel();
  test_memcpy_kernel();
  test_access_wrapper();
  test_constant_memory();
  test_global_memory();
  test_shared_memory();
  test_local_memory();

  sycl::queue q;
  test_mempcy_pitched(q);

  return 0;
}
