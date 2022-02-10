// REQUIRES: opencl-aot, accelerator

// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Check that FPGA emulator device is found if we try to initialize inline
// global variable using fpga_emulator_selector parameter.

inline cl::sycl::queue fpga_emu_queue_inlined{
    cl::sycl::ext::intel::fpga_emulator_selector{}};

int main() { return 0; }
