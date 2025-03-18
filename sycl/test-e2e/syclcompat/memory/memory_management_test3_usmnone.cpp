// ====------ memory_management_test3_usmnone.cpp---------- -*- C++ -* ----===////
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

void test_mempcy_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int N1 = 1000;
  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = (float *)syclcompat::malloc(Num * sizeof(float));
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, N1 * sizeof(float));
  syclcompat::memcpy_async((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float));
  syclcompat::memcpy_async((void*) h_C, (void*) d_A, Num * sizeof(float));

  syclcompat::wait_and_free((void*)d_A);

  syclcompat::free(0);
  syclcompat::free(NULL);
  syclcompat::free(nullptr);

  // verify
  for(int i = 0; i < N1; i++){
      if (fabs(h_A[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  for(int i = N1; i < Num; i++){
      if (fabs(h_B[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_buffer_and_offset_kernel() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int Offset = 0; // Current dpcpp version in ics environment has bugs with Offset > 0,
                  // CORC-6222 has fixed this issue, but the version of dpcpp used in ics
                  // environment has not cover this patch. After it has this patch,
                  // Offest could be set to 100, and current test case will pass.

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  //syclcompat::dev_mgr::instance().select_device(0);

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
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, Num * sizeof(float));
  syclcompat::memcpy_async((void*) d_B, (void*) h_B, Num * sizeof(float));

  syclcompat::get_default_queue().wait_and_throw();

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_A = syclcompat::get_buffer_and_offset(d_A);
    size_t offset_A = buffer_and_offset_A.second;
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_B = syclcompat::get_buffer_and_offset(d_B);
    size_t offset_B = buffer_and_offset_A.second;
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_C = syclcompat::get_buffer_and_offset(d_C);
    size_t offset_C = buffer_and_offset_A.second;
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto d_A_acc = buffer_and_offset_A.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_B_acc = buffer_and_offset_B.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_C_acc = buffer_and_offset_C.first.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class vectorAdd3_1>(
          sycl::range<1>(Num-Offset),
          [=](sycl::id<1> id) {

            float *A = (float*)(&d_A_acc[0]+offset_A);
            float *B = (float*)(&d_B_acc[0]+offset_B);
            float *C = (float*)(&d_C_acc[0]+offset_C);
             int i = id[0];

            C[i] = A[i] + B[i];
          });
      });
  }
  syclcompat::memcpy_async((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float));

  syclcompat::get_default_queue().wait_and_throw();

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


void test_memset_async() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 10;
  int *h_A = (int*)malloc(Num*sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A;

  d_A = (int *)syclcompat::malloc(Num * sizeof(int));
  // hostA -> deviceA
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, Num * sizeof(int));

  // set d_A[0,..., 6] = 0
  syclcompat::memset_async((void*) d_A, 0, (Num - 3) * sizeof(int));

  // deviceA -> hostA
  syclcompat::memcpy_async((void*) h_A, (void*) d_A, Num * sizeof(int));

  syclcompat::get_default_queue().wait_and_throw();

  syclcompat::free((void*)d_A);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    if (h_A[i] != 0) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element [%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    if (h_A[i] != 4) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element h_A[%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  free(h_A);
}

const unsigned int Num = 5000;
const unsigned int N1 = 1000;
syclcompat::constant_memory<float, 1> d_A(Num * sizeof(float));
syclcompat::constant_memory<float, 1> d_B(Num * sizeof(float));

void test_memcpy_async_getptr() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  float h_A[Num];
  float h_B[Num];
  float h_C[Num];
  float h_D[Num];

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> deviceB[0..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  // deviceB[0..4999] -> hostD[0..4999]
  syclcompat::memcpy_async((void *)d_A.get_ptr(), (void *)&h_A[0], N1 * sizeof(float));
  syclcompat::memcpy_async((char *)d_A.get_ptr() + N1 * sizeof(float), (void*) h_B, (Num-N1) * sizeof(float));
  syclcompat::memcpy_async((void *)h_C, (void *)d_A.get_ptr(), Num * sizeof(float));
  syclcompat::memcpy_async((void *)d_B.get_ptr(), (void *)d_A.get_ptr(), N1 * sizeof(float));
  syclcompat::memcpy_async((char *)d_B.get_ptr() + N1 * sizeof(float), (void *)((size_t)d_A.get_ptr() + N1* sizeof(float)), (Num - N1) * sizeof(float));
  syclcompat::memcpy_async((void *)h_D, (void *)d_B.get_ptr(), Num * sizeof(float));

  syclcompat::get_default_queue().wait_and_throw();
  // verify hostD
  for (int i = 0; i < N1; i++) {
    if (fabs(h_A[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are A = %f, D = %f:\n", h_A[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }

  for (int i = N1; i < Num; i++) {
    if (fabs(h_B[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are B = %f, D = %f:\n",   h_B[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }
}

void test_memcpy_pitched_async() {
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
  syclcompat::memcpy_async(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height);

  // copy back to host.
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height);

  syclcompat::get_default_queue().wait_and_throw();
  check(h_data, h_ref, width, height, 1);

  // memset device data.
  syclcompat::memset_async(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height);
  syclcompat::get_default_queue().wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data);
}

void test_mempcy_async(sycl::queue &q) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 5000;
  int N1 = 1000;
  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A;
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  d_A = (float *)syclcompat::malloc(Num * sizeof(float), q);
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, N1 * sizeof(float), q);
  syclcompat::memcpy_async((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), q);
  syclcompat::memcpy_async((void*) h_C, (void*) d_A, Num * sizeof(float), q);
  q.wait_and_throw();
  syclcompat::free((void*)d_A, q);

  syclcompat::free(0, q);
  syclcompat::free(NULL, q);
  syclcompat::free(nullptr, q);

  // verify
  for(int i = 0; i < N1; i++){
      if (fabs(h_A[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  for(int i = N1; i < Num; i++){
      if (fabs(h_B[i] - h_C[i]) > 1e-5) {
          fprintf(stderr,"Check: Elements are A = %f, B = %f, C = %f:\n", h_A[i],  h_B[i],  h_C[i]);
          fprintf(stderr,"Result verification failed at element %d:\n", i);
          exit(EXIT_FAILURE);
      }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

void test_buffer_and_offset_kernel(sycl::queue &q) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  int Num = 5000;
  int Offset = 0; // Current dpcpp version in ics environment has bugs with Offset > 0,
                  // CORC-6222 has fixed this issue, but the version of dpcpp used in ics
                  // environment has not cover this patch. After it has this patch,
                  // Offest could be set to 100, and current test case will pass.

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  //syclcompat::dev_mgr::instance().select_device(0);

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  d_A = (float *)syclcompat::malloc(Num * sizeof(float), q);
  d_B = (float *)syclcompat::malloc(Num * sizeof(float), q);
  d_C = (float *)syclcompat::malloc(Num * sizeof(float), q);
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, Num * sizeof(float), q);
  syclcompat::memcpy_async((void*) d_B, (void*) h_B, Num * sizeof(float), q);
  q.wait_and_throw();
  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_A = syclcompat::get_buffer_and_offset(d_A);
    size_t offset_A = buffer_and_offset_A.second;
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_B = syclcompat::get_buffer_and_offset(d_B);
    size_t offset_B = buffer_and_offset_A.second;
    std::pair<syclcompat::buffer_t, size_t> buffer_and_offset_C = syclcompat::get_buffer_and_offset(d_C);
    size_t offset_C = buffer_and_offset_A.second;
    syclcompat::get_default_queue().submit(
      [&](sycl::handler &cgh) {
      auto d_A_acc = buffer_and_offset_A.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_B_acc = buffer_and_offset_B.first.get_access<sycl::access::mode::read_write>(cgh);
      auto d_C_acc = buffer_and_offset_C.first.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class vectorAdd3_2>(
          sycl::range<1>(Num-Offset),
          [=](sycl::id<1> id) {

            float *A = (float*)(&d_A_acc[0]+offset_A);
            float *B = (float*)(&d_B_acc[0]+offset_B);
            float *C = (float*)(&d_C_acc[0]+offset_C);
             int i = id[0];

            C[i] = A[i] + B[i];
          });
      });
  }
  syclcompat::memcpy_async((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float), q);
  q.wait_and_throw();
  syclcompat::free((void*)d_A, q);
  syclcompat::free((void*)d_B, q);
  syclcompat::free((void*)d_C, q);

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


void test_memset_async(sycl::queue &q) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  int Num = 10;
  int *h_A = (int*)malloc(Num*sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A;

  d_A = (int *)syclcompat::malloc(Num * sizeof(int), q);
  // hostA -> deviceA
  syclcompat::memcpy_async((void*) d_A, (void*) h_A, Num * sizeof(int), q);

  // set d_A[0,..., 6] = 0
  syclcompat::memset_async((void*) d_A, 0, (Num - 3) * sizeof(int), q);

  // deviceA -> hostA
  syclcompat::memcpy_async((void*) h_A, (void*) d_A, Num * sizeof(int), q);
  syclcompat::wait_and_free((void*)d_A, q);

  // check d_A[0,..., 6] = 0
  for (int i = 0; i < Num - 3; i++) {
    if (h_A[i] != 0) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element [%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // check d_A[7,..., 9] = 4
  for (int i = Num - 3; i < Num; i++) {
    if (h_A[i] != 4) {
      fprintf(stderr, "Check: h_A[%d] is %d:\n", i, h_A[i]);
      fprintf(stderr, "Result verification failed at element h_A[%d]!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  free(h_A);
}

void test_memcpy_async_getptr(sycl::queue &q) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;

  const unsigned int Num = 5000;
  const unsigned int N1 = 1000;
  syclcompat::constant_memory<float, 1> d_A(Num * sizeof(float));
  syclcompat::constant_memory<float, 1> d_B(Num * sizeof(float));

  float h_A[Num];
  float h_B[Num];
  float h_C[Num];
  float h_D[Num];

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  // hostA[0..999] -> deviceA[0..999]
  // hostB[0..3999] -> deviceA[1000..4999]
  // deviceA[0..4999] -> deviceB[0..4999]
  // deviceA[0..4999] -> hostC[0..4999]
  // deviceB[0..4999] -> hostD[0..4999]
  syclcompat::memcpy_async((void *)d_A.get_ptr(), (void *)&h_A[0], N1 * sizeof(float), q);
  syclcompat::memcpy_async((char *)d_A.get_ptr() + N1 * sizeof(float), (void*) h_B, (Num-N1) * sizeof(float), q);
  syclcompat::memcpy_async((void *)h_C, (void *)d_A.get_ptr(), Num * sizeof(float), q);
  syclcompat::memcpy_async((void *)d_B.get_ptr(), (void *)d_A.get_ptr(), N1 * sizeof(float), q);
  syclcompat::memcpy_async((char *)d_B.get_ptr() + N1 * sizeof(float), (void *)((size_t)d_A.get_ptr() + N1* sizeof(float)), (Num - N1) * sizeof(float), q);
  syclcompat::memcpy_async((void *)h_D, (void *)d_B.get_ptr(), Num * sizeof(float), q);
  q.wait_and_throw();
  // verify hostD
  for (int i = 0; i < N1; i++) {
    if (fabs(h_A[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are A = %f, D = %f:\n", h_A[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }

  for (int i = N1; i < Num; i++) {
    if (fabs(h_B[i] - h_D[i]) > 1e-5) {
      fprintf(stderr, "Check: Elements are B = %f, D = %f:\n",   h_B[i], h_D[i]);
      fprintf(stderr, "Result verification failed at element %d:\n", i);
      exit(EXIT_FAILURE);
    }
  }
}

void test_memcpy_pitched_async(sycl::queue &q) {
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
  syclcompat::memcpy_async(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height, q);

  // copy back to host.
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, q);
  q.wait_and_throw();
  check(h_data, h_ref, width, height, 1);

  // memset device data.
  syclcompat::memset_async(d_data, d_pitch, 0x1, sizeof(float) * width, height, q);

  // copy back to host
  syclcompat::memcpy_async(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, q);
  q.wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  syclcompat::free((void *)d_data, q);
}

int main() {
  test_mempcy_async();
  test_buffer_and_offset_kernel();
  test_memset_async();
  test_memcpy_async_getptr();
  test_memcpy_pitched_async();

  sycl::queue q;
  test_mempcy_async(q);
  test_buffer_and_offset_kernel(q);
  test_memset_async(q);
  test_memcpy_async_getptr(q);
  test_memcpy_pitched_async(q);

  return 0;
}
