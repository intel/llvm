// ====------ memory_management_test3.cpp---------- -*- C++ -* ----===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===----------------------------------------------------------------------===//


#define DPCT_USM_LEVEL_NONE
#include <sycl/sycl.hpp>
#include <dpct/memory.hpp>

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

void test1() {

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
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float));
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), dpct::host_to_device);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), dpct::host_to_device);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), dpct::device_to_host);

  dpct::get_default_queue().wait_and_throw();

  dpct::dpct_free((void*)d_A);

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

  printf("Test1 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

void test2() {

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
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float));
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), dpct::automatic);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), dpct::automatic);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), dpct::automatic);

  dpct::get_default_queue().wait_and_throw();

  dpct::dpct_free((void*)d_A);

  dpct::dpct_free(0);
  dpct::dpct_free(NULL);
  dpct::dpct_free(nullptr);

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

  printf("Test2 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

class vectorAdd3;
void test3() {

  int Num = 5000;
  int Offset = 0; // Current dpcpp version in ics environment has bugs with Offset > 0,
                  // CORC-6222 has fixed this issue, but the version of dpcpp used in ics
                  // environment has not cover this patch. After it has this patch,
                  // Offest could be set to 100, and current test case will pass.

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  //dpct::dev_mgr::instance().select_device(0);

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float));
  d_B = (float *)dpct::dpct_malloc(Num * sizeof(float));
  d_C = (float *)dpct::dpct_malloc(Num * sizeof(float));
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, Num * sizeof(float), dpct::host_to_device);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_B, (void*) h_B, Num * sizeof(float), dpct::host_to_device);

  dpct::get_default_queue().wait_and_throw();

  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_A = dpct::get_buffer_and_offset(d_A);
    size_t offset_A = buffer_and_offset_A.second;
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_B = dpct::get_buffer_and_offset(d_B);
    size_t offset_B = buffer_and_offset_A.second;
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_C = dpct::get_buffer_and_offset(d_C);
    size_t offset_C = buffer_and_offset_A.second;
    dpct::get_default_queue().submit(
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
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float), dpct::device_to_host);

  dpct::get_default_queue().wait_and_throw();

  dpct::dpct_free((void*)d_A);
  dpct::dpct_free((void*)d_B);
  dpct::dpct_free((void*)d_C);

  // verify
  for(int i = Offset; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f, C = %f:\n", i,h_A[i],  h_B[i],  h_C[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  printf("Test3 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}


void test4() {

  int Num = 10;
  int *h_A = (int*)malloc(Num*sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A;

  d_A = (int *)dpct::dpct_malloc(Num * sizeof(int));
  // hostA -> deviceA
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, Num * sizeof(int), dpct::host_to_device);

  // set d_A[0,..., 6] = 0
  //test_feature:async_dpct_memset
  dpct::async_dpct_memset((void*) d_A, 0, (Num - 3) * sizeof(int));

  // deviceA -> hostA
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_A, (void*) d_A, Num * sizeof(int), dpct::device_to_host);

  dpct::get_default_queue().wait_and_throw();

  dpct::dpct_free((void*)d_A);

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

  printf("Test4 Passed\n");

  free(h_A);
}

const unsigned int Num = 5000;
const unsigned int N1 = 1000;
dpct::constant_memory<float, 1> d_A(Num * sizeof(float));
dpct::constant_memory<float, 1> d_B(Num * sizeof(float));

void test5() {

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
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)d_A.get_ptr(), (void *)&h_A[0], N1 * sizeof(float), dpct::host_to_device);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((char *)d_A.get_ptr() + N1 * sizeof(float), (void*) h_B, (Num-N1) * sizeof(float), dpct::automatic);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)h_C, (void *)d_A.get_ptr(), Num * sizeof(float),   dpct::device_to_host);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)d_B.get_ptr(), (void *)d_A.get_ptr(), N1 * sizeof(float), dpct::device_to_device);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((char *)d_B.get_ptr() + N1 * sizeof(float), (void *)((size_t)d_A.get_ptr() + N1* sizeof(float)), (Num - N1) * sizeof(float), dpct::automatic);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)h_D, (void *)d_B.get_ptr(), Num * sizeof(float),   dpct::device_to_host);

  dpct::get_default_queue().wait_and_throw();
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

  printf("Test5 Passed\n");
}

void test6() {
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
  d_data = (float *)dpct::dpct_malloc(d_pitch, sizeof(float) * width, height);

  // copy to Device.
  dpct::memcpy_direction cpyDir = dpct::host_to_device;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height, cpyDir);

  // copy back to host.
  cpyDir = dpct::device_to_host;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, cpyDir);

  dpct::get_default_queue().wait_and_throw();
  check(h_data, h_ref, width, height, 1);

  // memset device data.
  //test_feature:async_dpct_memset
  dpct::async_dpct_memset(d_data, d_pitch, 0x1, sizeof(float) * width, height);

  // copy back to host
  cpyDir = dpct::device_to_host;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, cpyDir);
  dpct::get_default_queue().wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  dpct::dpct_free((void *)d_data);

  printf("Test6 passed!\n");
}

void test1(sycl::queue &q) {

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
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float), q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), dpct::host_to_device, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), dpct::host_to_device), q;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), dpct::device_to_host, q);
  q.wait_and_throw();
  dpct::dpct_free((void*)d_A, q);

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

  printf("Test1 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

void test2(sycl::queue &q) {

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
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float), q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, N1 * sizeof(float), dpct::automatic, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (d_A + N1), (void*) h_B, (Num-N1) * sizeof(float), dpct::automatic, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_C, (void*) d_A, Num * sizeof(float), dpct::automatic, q);
  q.wait_and_throw();
  dpct::dpct_free((void*)d_A, q);

  dpct::dpct_free(0, q);
  dpct::dpct_free(NULL, q);
  dpct::dpct_free(nullptr, q);

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

  printf("Test2 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}

void test3(sycl::queue &q) {
  class vectorAdd3;
  int Num = 5000;
  int Offset = 0; // Current dpcpp version in ics environment has bugs with Offset > 0,
                  // CORC-6222 has fixed this issue, but the version of dpcpp used in ics
                  // environment has not cover this patch. After it has this patch,
                  // Offest could be set to 100, and current test case will pass.

  float *h_A = (float*)malloc(Num*sizeof(float));
  float *h_B = (float*)malloc(Num*sizeof(float));
  float *h_C = (float*)malloc(Num*sizeof(float));

  //dpct::dev_mgr::instance().select_device(0);

  for (int i = 0; i < Num; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  float *d_A, *d_B, *d_C;
  // hostA -> deviceA
  // hostB -> deviceB
  // kernel: deviceC = deviceA + deviceB
  // deviceA -> hostC
  d_A = (float *)dpct::dpct_malloc(Num * sizeof(float), q);
  d_B = (float *)dpct::dpct_malloc(Num * sizeof(float), q);
  d_C = (float *)dpct::dpct_malloc(Num * sizeof(float), q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, Num * sizeof(float), dpct::host_to_device, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_B, (void*) h_B, Num * sizeof(float), dpct::host_to_device, q);
  q.wait_and_throw();
  d_A += Offset;
  d_B += Offset;
  d_C += Offset;

  {
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_A = dpct::get_buffer_and_offset(d_A);
    size_t offset_A = buffer_and_offset_A.second;
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_B = dpct::get_buffer_and_offset(d_B);
    size_t offset_B = buffer_and_offset_A.second;
    std::pair<dpct::buffer_t, size_t> buffer_and_offset_C = dpct::get_buffer_and_offset(d_C);
    size_t offset_C = buffer_and_offset_A.second;
    dpct::get_default_queue().submit(
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
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) (h_C+Offset), (void*) d_C, (Num-Offset) * sizeof(float), dpct::device_to_host, q);
  q.wait_and_throw();
  dpct::dpct_free((void*)d_A, q);
  dpct::dpct_free((void*)d_B, q);
  dpct::dpct_free((void*)d_C, q);

  // verify
  for(int i = Offset; i < Num; i++){
      if (fabs(h_C[i] - h_A[i] - h_B[i]) > 1e-5) {
        fprintf(stderr,"Check %d: Elements are A = %f, B = %f, C = %f:\n", i,h_A[i],  h_B[i],  h_C[i]);
        fprintf(stderr,"Result verification failed at element %d:\n", i);
        exit(EXIT_FAILURE);
      }
  }

  printf("Test3 Passed\n");

  free(h_A);
  free(h_B);
  free(h_C);
}


void test4(sycl::queue &q) {

  int Num = 10;
  int *h_A = (int*)malloc(Num*sizeof(int));

  for (int i = 0; i < Num; i++) {
    h_A[i] = 4;
  }

  int *d_A;

  d_A = (int *)dpct::dpct_malloc(Num * sizeof(int), q);
  // hostA -> deviceA
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) d_A, (void*) h_A, Num * sizeof(int), dpct::host_to_device, q);

  // set d_A[0,..., 6] = 0
  //test_feature:async_dpct_memset
  dpct::async_dpct_memset((void*) d_A, 0, (Num - 3) * sizeof(int), q);

  // deviceA -> hostA
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void*) h_A, (void*) d_A, Num * sizeof(int), dpct::device_to_host, q);
  q.wait_and_throw();
  dpct::dpct_free((void*)d_A, q);

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

  printf("Test4 Passed\n");

  free(h_A);
}

void test5(sycl::queue &q) {

  const unsigned int Num = 5000;
  const unsigned int N1 = 1000;
  dpct::constant_memory<float, 1> d_A(Num * sizeof(float));
  dpct::constant_memory<float, 1> d_B(Num * sizeof(float));

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
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)d_A.get_ptr(), (void *)&h_A[0], N1 * sizeof(float), dpct::host_to_device, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((char *)d_A.get_ptr() + N1 * sizeof(float), (void*) h_B, (Num-N1) * sizeof(float), dpct::automatic, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)h_C, (void *)d_A.get_ptr(), Num * sizeof(float),   dpct::device_to_host, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)d_B.get_ptr(), (void *)d_A.get_ptr(), N1 * sizeof(float), dpct::device_to_device, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((char *)d_B.get_ptr() + N1 * sizeof(float), (void *)((size_t)d_A.get_ptr() + N1* sizeof(float)), (Num - N1) * sizeof(float), dpct::automatic, q);
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy((void *)h_D, (void *)d_B.get_ptr(), Num * sizeof(float),   dpct::device_to_host, q);
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

  printf("Test5 Passed\n");
}

void test6(sycl::queue &q) {
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
  d_data = (float *)dpct::dpct_malloc(d_pitch, sizeof(float) * width, height, q);

  // copy to Device.
  dpct::memcpy_direction cpyDir = dpct::host_to_device;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(d_data, d_pitch, h_data, h_pitch, sizeof(float) * width, height, cpyDir, q);

  // copy back to host.
  cpyDir = dpct::device_to_host;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, cpyDir, q);
  q.wait_and_throw();
  check(h_data, h_ref, width, height, 1);

  // memset device data.
  //test_feature:async_dpct_memset
  dpct::async_dpct_memset(d_data, d_pitch, 0x1, sizeof(float) * width, height, q);

  // copy back to host
  cpyDir = dpct::device_to_host;
  //test_feature:async_dpct_memcpy
  dpct::async_dpct_memcpy(h_data, h_pitch, d_data, d_pitch, sizeof(float) * width, height, cpyDir, q);
  q.wait_and_throw();
  // memset reference data.
  memset(h_ref, 0x1, width * height * sizeof(float));
  check(h_data, h_ref, width, height, 1);

  free(h_data);
  free(h_ref);
  dpct::dpct_free((void *)d_data, q);

  printf("Test6 passed!\n");
}

int main() {
  test1();
  test2();
  test3();
  test4();
  test5();
  test6();

  sycl::queue q;
  test1(q);
  test2(q);
  test3(q);
  test4(q);
  test5(q);
  test6(q);

  return 0;
}
