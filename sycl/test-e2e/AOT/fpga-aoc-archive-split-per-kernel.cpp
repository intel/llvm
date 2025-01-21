// Test compiler behaviors for -fintelfpga with -fsycl-link with split
// per-kernel.

// REQUIRES: opencl-aot, accelerator

// Remove any archives
// RUN: rm -f %t_*.a

// Build main object.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -c %S/Inputs/fpga_main.cpp -o %t_main.o

// Build any early archive binaries.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=early %S/Inputs/fpga_sub.cpp   -o %t_early_sub.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=early %S/Inputs/fpga_add.cpp   -o %t_early_add.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=early %S/Inputs/fpga_sub_x.cpp -o %t_early_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=early %S/Inputs/fpga_add_x.cpp -o %t_early_add_x.a

// Test baseline of all early archives and main.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl %t_main.o %t_early_add.a %t_early_sub.a %t_early_add_x.a %t_early_sub_x.a -o %t_early.out
// RUN: %{run} %t_early.out

// Build any image archive binaries.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %S/Inputs/fpga_sub.cpp   -o %t_image_sub.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %S/Inputs/fpga_add.cpp   -o %t_image_add.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %S/Inputs/fpga_sub_x.cpp -o %t_image_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %S/Inputs/fpga_add_x.cpp -o %t_image_add_x.a

// Test baseline of all image archives and main.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl %t_main.o %t_image_add.a %t_image_sub.a %t_image_add_x.a %t_image_sub_x.a -o %t_image.out
// RUN: %{run} %t_image.out

// Build any image archive binaries from early archives.
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %t_early_sub_x.a -o %t_early_image_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl -fsycl-link=image %t_early_add_x.a -o %t_early_image_add_x.a

// Mix early and image archive usage
// RUN: %clangxx -fintelfpga -fsycl-device-code-split=per_kernel -fsycl %t_main.o %t_early_add.a %t_image_sub.a %t_early_image_add_x.a %t_early_image_sub_x.a -o %t_mix.out
// RUN: %{run} %t_mix.out
