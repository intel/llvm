// REQUIRES: opencl-aot, accelerator

// RUN: %clangxx -fsycl -fintelfpga %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// Check that FPGA emulator device is found if we try to initialize inline
// global variable using fpga_emulator_selector parameter.

inline sycl::queue fpga_emu_queue_inlined{
    sycl::ext::intel::fpga_emulator_selector_v};

int main() { return 0; }
