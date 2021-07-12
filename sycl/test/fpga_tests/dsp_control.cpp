// RUN: %clangxx -fsycl-device-only -Xclang -fsycl-is-device -emit-llvm %s -S -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// Check DSP control interface

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_dsp_control.hpp>
#include <iostream>

using namespace sycl;
constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

void RunKernel(queue &deviceQueue, const std::vector<float> &in,
               std::vector<float> &out) {

  buffer<float, 1> in_buf(in);
  buffer<float, 1> out_buf(out);

  deviceQueue.submit([&](handler &cgh) {
    auto in_acc = in_buf.template get_access<sycl_read>(cgh);
    auto out_acc = out_buf.template get_access<sycl_write>(cgh);

    cgh.single_task<class Kernel>([=] {
      float sum = in_acc[0];
      sum += 1.23f;
      
      // CHECK: define {{.*}}spir_func void @{{.*}}math_prefer_softlogic_propagate{{.*}} !prefer_dsp [[MD_i32_0:![0-9]+]] !propagate_dsp_preference [[MD_i32_1:![0-9]+]]
      INTEL::math_dsp_control<INTEL::Preference::Softlogic>(
          [&] { sum += 4.56f; });
      
      // CHECK: define {{.*}}spir_func void @{{.*}}math_prefer_dsp_propagate{{.*}} !prefer_dsp [[MD_i32_1:![0-9]+]] !propagate_dsp_preference [[MD_i32_1:![0-9]+]]
      INTEL::math_dsp_control<INTEL::Preference::DSP>([&] { sum += 4.56f; });
      
      // CHECK: define {{.*}}spir_func void @{{.*}}math_prefer_softlogic_no_propagate{{.*}} !prefer_dsp [[MD_i32_0:![0-9]+]]
      INTEL::math_dsp_control<INTEL::Preference::Softlogic,
                              INTEL::Propagate::Off>([&] { sum += 4.56f; });
      
      // CHECK: define {{.*}}spir_func void @{{.*}}math_prefer_dsp_no_propagate{{.*}} !prefer_dsp [[MD_i32_1:![0-9]+]]
      INTEL::math_dsp_control<INTEL::Preference::DSP, INTEL::Propagate::Off>(
          [&] { sum += 4.56f; });
      out_acc[0] = sum;
    });
  });
}

int main() {
  std::vector<float> in{0.0f};
  std::vector<float> out{0.0f};
  queue deviceQueue(testconfig_selector{}, &m_exception_handler);
  RunKernel(deviceQueue, in, out);

  float golden = 19.47f;
  if (fabs(out[0] - golden) < (out[0] * 1e-5)) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
    std::cout << "result = " << out[0] << ", golden = " << golden << "\n";
  }
  return 0;
}

// CHECK: [[MD_i32_0]] = !{i32 0}
// CHECK: [[MD_i32_1]] = !{i32 1}
