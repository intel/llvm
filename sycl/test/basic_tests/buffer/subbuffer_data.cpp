// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 SYCL_DEVICE_TYPE=HOST %t.out
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_PI_TRACE=2 %ACC_RUN_PLACEHOLDER %t.out
//
//==---------- subbuffer_data.cpp --- sub-buffer test for data copy back and destruction ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
using namespace cl::sycl;

constexpr long total = 1'024;
constexpr long subSz = total / 2;

// setup_arr makes data go 1, 2, ... n. (like iota)
void setup_arr(int *arr) {
  for (long i = 0; i < total; i++) {
    arr[i] = i;
  }
}

// clear_arr sets all data to 0.
void clear_arr(int *arr, int offset = 0) {
  for (long i = offset; i < total; i++) {
    arr[i] = 0;
  }
}

void report_arr(int *arr, std::string msg) {
  std::cout << ":: " << msg << " ::" << std::endl;
  for (long i = 0; i < total; i++) {
    std::cout << i << ": " << arr[i] << std::endl;
  }
}

void ensureNoUnecessaryCopyBack(queue &q) {

  std::cout << "start ensureNoUnecessaryCopyBack" << std::endl;

  //allocate  memory
  int *baseData = (int *)(malloc(total * sizeof(int)));
  int *otherData = (int *)(malloc(total * sizeof(int)));

  // ------- basic copy back ------------
  //  two buffers are setup, one is read from, one is written to.
  //  When ~dtor, only the written-to buffer should have been copied back.
  //  We test this by clearing the read buffer behind its back and then check its value after ~dtor,
  //  to make sure that it was not copied back
  //  SYCL guarantees that the write buffer will have copied back no later than its ~dtor, but, in fact,
  //  it might copy back earlier than that (and almost certainly will for CPU bound devices).
  //  But we don't care about the write buffer. We only care that the read buffer was NOT copied-back.
  { // closure
    //setup and clear memory
    setup_arr(baseData);  // [0, 1, 2,  ..., total]
    clear_arr(otherData); // [0, 0, 0, ..., 0]
    //buffers
    buffer<int, 1> readFrom(baseData, range<1>(total));
    buffer<int, 1> writeTo(otherData, range<1>(total));

    q.submit([&](handler &cgh) {
      auto readAcc = readFrom.get_access<access::mode::read>(cgh);
      auto writeAcc = writeTo.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class noCopyBackRead>(range<1>(total), [=](id<1> i) {
        writeAcc[i] = readAcc[i] + 1;
      });
    });
    q.wait();

    // change readBuffer behind its back.
    clear_arr(baseData);
  }                                                                                                 // end closure. Buffers ~dtor and copied back to backing data.  Only the 'otherData' should have changed.
  assert(otherData[1] == 2 && "otherData should have been updated no later than ~dtor copy-back."); // trivial. Tripping this indicates some unknown problem.
  assert(baseData[1] == 0 && "baseData should NOT have been copy-back by ~dtor.");                  // this is our main test. Ensuring the read buffer was NOT copied-back.

  free(baseData);
  free(otherData);

  std::cout << "end ensureNoUnecessaryCopyBack" << std::endl;
}
// if there was only one copy-back, it will mean there is only one enqueuedMemBufferRead.
//CHECK: start ensureNoUnecessaryCopyBack
//CHECK: ---> piEnqueueMemBufferRead(
//CHECK-NOT: ---> piEnqueueMemBufferRead(
//CHECK: end ensureNoUnecessaryCopyBack

void ensureSubBufferDtorCopyBack(queue q) {

  std::cout << "start ensureSubBufferDtorCopyBack" << std::endl;

  //allocate and setup memory
  int *baseData = (int *)(malloc(total * sizeof(int)));
  setup_arr(baseData); // [0, 1, 2, ..., total]

  // -------- sub buffer copy back on ~dtor -----------
  //  per the SYCL spec, all buffers are expected to copy back data at/before their destruction.
  //  Here we use an extra closure so that a sub-buffer ~dtor executes before the base ~dtor.
  //
  { // closure

    // buffers
    buffer<int, 1> base(baseData, range<1>(total));
    { // closure for sub-buffer
      buffer<int, 1> subBuff(base, id<1>(0), range<1>(subSz));

      q.submit([&](handler &cgh) {
        auto subBuffAcc = subBuff.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class subDtorCopyBack>(range<1>(subSz), [=](id<1> i) {
          subBuffAcc[i] = subBuffAcc[i] * 2;
        });
      });
      q.wait();

    } // end sub-buffer closure.
    assert(baseData[1] == 2 && "the part of buffer covered by the sub-buffer should have copied back no later than now");

  } //end closure

  free(baseData);

  std::cout << "end ensureSubBufferDtorCopyBack" << std::endl;
}
// in addition to verifying that at ~dtor time the data was copied back,
// we also want to verify that ONLY the sub-buffer resulted in data copied back.
// that the base buffer did not also copy back data.
//CHECK: start ensureSubBufferDtorCopyBack
//CHECK: ---> piEnqueueMemBufferRead(
//CHECK-NOT: ---> piEnqueueMemBufferRead(
//CHECK: end ensureSubBufferDtorCopyBack

int do_nothing(int x) {
  return x + 1;
}
void ensureSubBufferReadDoesNOTCopyBack(queue q) {

  std::cout << "start ensureSubBufferReadDoesNOTCopyBack" << std::endl;

  //allocate and setup memory
  int *baseData = (int *)(malloc(total * sizeof(int)));
  setup_arr(baseData); // [0, 1, 2, 3, ..., total]

  // -------- sub buffer read only does NOT copy back on ~dtor -----------
  //
  { // closure

    // buffers
    buffer<int, 1> base(baseData, range<1>(total));
    { // closure for sub-buffer
      buffer<int, 1> subBuff(base, id<1>(0), range<1>(subSz));

      q.submit([&](handler &cgh) {
        auto subBuffAcc = subBuff.get_access<access::mode::read>(cgh);
        cgh.parallel_for<class subBufferReadNoCopyBack>(range<1>(subSz), [=](id<1> i) {
          int y = do_nothing(subBuffAcc[i]);
        });
      });
      q.wait();

      //change array behind back
      clear_arr(baseData);
    } // end sub-buffer closure.
    assert(baseData[1] == 0 && "the sub-buffer should not have copied back");

  } //end closure

  free(baseData);

  std::cout << "end ensureSubBufferReadDoesNOTCopyBack" << std::endl;
}
//CHECK: start ensureSubBufferReadDoesNOTCopyBack
//CHECK-NOT: ---> piEnqueueMemBufferRead(
//CHECK: end ensureSubBufferReadDoesNOTCopyBack

void checkSubSetWriteBack(queue q) {
  //calling set_write_back(false) should prevent sub-buffer from writing back.
  std::cout << "start checkSubSetWriteBack" << std::endl;

  //allocate and setup memory
  int *baseData = (int *)(malloc(total * sizeof(int)));
  setup_arr(baseData); // [0, 1, 2, 3, ..., total]
  {
    buffer<int, 1> base(baseData, range<1>(total));
    buffer<int, 1> subBuff(base, id<1>(0), range<1>(subSz));
    subBuff.set_write_back(false);
    q.submit([&](handler &cgh) {
      auto subBuffAcc = subBuff.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class checkSubSetWriteBack>(range<1>(subSz), [=](id<1> i) {
        subBuffAcc[i] = subBuffAcc[i] * 2;
      });
    });
  } //closure

  std::cout << "end checkSubSetWriteBack" << std::endl;
  free(baseData);
}
//CHECK: start checkSubSetWriteBack
//CHECK-NOT: ---> piEnqueueMemBufferRead(
//CHECK: end checkSubSetWriteBack

int main() {
  queue q;
  ensureNoUnecessaryCopyBack(q);
  ensureSubBufferDtorCopyBack(q);
  ensureSubBufferReadDoesNOTCopyBack(q);
  checkSubSetWriteBack(q);
}
