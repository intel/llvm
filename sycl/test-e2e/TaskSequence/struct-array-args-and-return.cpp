//==-------- struct-array-args-and-return.cpp - DPC++ task_sequence --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aspect-ext_intel_fpga_task_sequence
// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out
#include "common.hpp"

class TaskSequenceTest;

constexpr int kInputSize = 12;

template <typename T, int COUNT> class FixedVect {
  T data[COUNT];

public:
  T &operator[](int i) { return data[i]; }
};

struct FunctionPacket {
  float val;
  bool isValid;

  FunctionPacket() : val(-1.0f), isValid(false) {}
  FunctionPacket(float val_, bool isValid_) : val(val_), isValid(isValid_) {}
};

using T2_f = FixedVect<float, 2>;
using intertask_pipe1 = ext::intel::pipe<class p1, float, kInputSize>;
using intertask_pipe2 = ext::intel::pipe<class p2, float, kInputSize>;

template <typename OutPipe> void argStruct(FunctionPacket fp) {
  if (fp.isValid) {
    float val = fp.val;
    OutPipe::write(val * val);
  }
}

template <typename OutPipe> void argArray(T2_f t2_array) {
  for (int i = 0; i < 2; i++) {
    float val = t2_array[i];
    OutPipe::write(val * val);
  }
}

template <typename InPipe> FunctionPacket returnStruct(bool shouldRead) {
  FunctionPacket res{-1, false};
  if (shouldRead) {
    float a = InPipe::read();
    a = sqrt(a);
    res.val = a;
    res.isValid = true;
  }
  return res;
}

template <typename InPipe> T2_f returnArray(bool shouldRead) {
  T2_f res;
  for (int i = 0; i < 2; i++) {
    float a = -1;
    if (shouldRead) {
      a = InPipe::read();
      a = sqrt(a);
    }
    res[i] = a;
  }

  return res;
}

bool check(std::vector<T2_f> &res) {
  int golden = 0;
  bool passed = true;
  for (int i = 0; i < kInputSize; i++) {
    if (((i + 1) % 4) == 0) {
      if (res[i][0] != -1 || res[i][1] != -1) {
        std::cout << "output mismatch, expected -1\n";
        passed = false;
      }
    } else {
      if (res[i][0] != (float)golden || res[i][1] != (float)(golden + 1)) {
        std::cout << "output mismatch, expected: " << golden << " and "
                  << (golden + 1) << ", actual: " << res[i][0] << " and "
                  << res[i][1] << "\n";
        passed = false;
      }
      golden += 2;
    }
  }
  return passed;
}

bool check(std::vector<FunctionPacket> &res) {
  int golden = 0;
  bool passed = true;
  for (int i = 0; i < kInputSize; i++) {
    if (((i + 1) % 4) == 0) {
      if (res[i].isValid) {
        std::cout << "output mismatch, expected non-valid\n";
        passed = false;
      }
    } else {
      if (!res[i].isValid || res[i].val != (float)golden) {
        std::cout << "output mismatch, expected: " << golden
                  << ", actual: " << res[i].val << "\n";
        passed = false;
      }
      golden++;
    }
  }
  return passed;
}

int main() {
  std::vector<FunctionPacket> vec_in_struct(kInputSize);
  std::vector<T2_f> vec_in_array(kInputSize);
  std::vector<FunctionPacket> res_struct(kInputSize);
  std::vector<T2_f> res_array(kInputSize);

  for (int i = 0; i < kInputSize; i++) {
    vec_in_struct[i] = FunctionPacket{static_cast<float>(i), true};

    T2_f a;
    a[0] = static_cast<float>(2 * i);
    a[1] = static_cast<float>(2 * i + 1);
    vec_in_array[i] = a;
  }

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

        for (int i = 0; i < kInputSize; i++) {
          // launch struct task functions
          ts_func1Struct.async(in_acc_struct[i]);
          ts_func2Struct.async(((i + 1) % 4 != 0));

          // launch array task functions
          ts_func1Array.async(in_acc_array[i]);
          ts_func2Array.async(((i + 1) % 4 != 0));
        }

        for (int i = 0; i < kInputSize; i++) {
          out_acc_struct[i] = ts_func2Struct.get();
          out_acc_array[i] = ts_func2Array.get();
        }
      });
    });
    q.wait();
  }
  bool passed = check(res_struct) && check(res_array);
  std::cout << (passed ? "PASSED\n" : "FAILED\n");

  return 0;
}
