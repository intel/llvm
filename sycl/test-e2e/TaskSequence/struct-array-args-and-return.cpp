//==-------- struct-array-args-and-return.cpp - DPC++ task_sequence --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: compfail, see https://github.com/intel/llvm/issues/14284, re-enable
// when fixed:
// UNSUPPORTED: linux, windows

// REQUIRES: aspect-ext_intel_fpga_task_sequence
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out
#include "common.hpp"

class TaskSequenceTest;


template <typename T, int COUNT> class FixedVect {
  T data[COUNT];

public:
  T &operator[](int i) { return data[i]; }
};

struct DataStruct {
  float val;
  bool isValid;

  DataStruct() : val(-1.0f), isValid(false) {}
  DataStruct(float val_, bool isValid_) : val(val_), isValid(isValid_) {}
};

using DataArray = FixedVect<float, 2>;
using intertask_pipe1 = ext::intel::pipe<class p1, float, 1>;
using intertask_pipe2 = ext::intel::pipe<class p2, float, 1>;

template <typename OutPipe> void argStruct(DataStruct data) {
  if (data.isValid) {
    OutPipe::write(data.val * data.val);
  }
}

template <typename OutPipe> void argArray(DataArray data) {
  for (int i = 0; i < 2; i++) {
    float a = data[i];
    OutPipe::write(a * a);
  }
}

template <typename InPipe> DataStruct returnStruct() {
  float a = InPipe::read();
  DataStruct res{sycl::sqrt(a), true};
  return res;
}

template <typename InPipe> DataArray returnArray() {
  DataArray res;
  for (int i = 0; i < 2; i++) {
    float a = InPipe::read();
    res[i] = sycl::sqrt(a);
  }

  return res;
}

int main() {
  std::vector<DataStruct> vec_in_struct(1);
  std::vector<DataArray> vec_in_array(1);
  std::vector<DataStruct> res_struct(1);
  std::vector<DataArray> res_array(1);

  vec_in_struct[0] = DataStruct{5.0, true};

  DataArray d;
  d[0] = 7.0;
  d[1] = 9.0;
  vec_in_array[0] = d;

  {
    queue q;

    buffer buffer_in_struct(vec_in_struct);
    buffer buffer_in_array(vec_in_array);
    buffer buffer_out_struct(res_struct);
    buffer buffer_out_array(res_array);

    q.submit([&](handler &h) {
      accessor in_acc_struct(buffer_in_struct, h, read_only);
      accessor in_acc_array(buffer_in_array, h, read_only);
      accessor out_acc_struct(buffer_out_struct, h, write_only, no_init);
      accessor out_acc_array(buffer_out_array, h, write_only, no_init);
      h.single_task<TaskSequenceTest>([=]() {
        task_sequence<argStruct<intertask_pipe1>> ts_func1Struct;
        task_sequence<argArray<intertask_pipe2>> ts_func1Array;
        task_sequence<returnStruct<intertask_pipe1>> ts_func2Struct;
        task_sequence<returnArray<intertask_pipe2>> ts_func2Array;

        // launch struct task functions
        ts_func1Struct.async(in_acc_struct[0]);
        ts_func2Struct.async();

        // launch array task functions
        ts_func1Array.async(in_acc_array[0]);
        ts_func2Array.async();

        out_acc_struct[0] = ts_func2Struct.get();
        out_acc_array[0] = ts_func2Array.get();
      });
    });
    q.wait();
  }
  assert((std::abs(res_struct[0].val - vec_in_struct[0].val) < 0.001) &&
         res_struct[0].isValid);
  assert((std::abs(res_array[0][0] - vec_in_array[0][0]) < 0.001) &&
         (std::abs(res_array[0][1] - vec_in_array[0][1]) < 0.001));
  return 0;
}
