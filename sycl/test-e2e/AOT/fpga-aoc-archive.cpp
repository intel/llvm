// Test compiler behaviors for -fintelfpga with -fsycl-link.

// REQUIRES: opencl-aot, accelerator

// Build any early archive binaries.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=early %S/Inputs/fpga_sub.cpp -o %t_early_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=early %S/Inputs/fpga_add.cpp -o %t_early_add.a
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_early_add.a %t_early_sub.a -o %t_early.out
// RUN: %{run} %t_early.out

// Build any image archive binaries.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_sub.cpp -o %t_image_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_add.cpp -o %t_image_add.a
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_image_add.a %t_image_sub.a -o %t_image.out
// RUN: %{run} %t_image.out

// Build any image archive binaries from early archives.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %t_early_sub.a -o %t_early_image_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %t_early_add.a -o %t_early_image_add.a
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_early_image_add.a %t_early_image_sub.a -o %t_early_image.out
// RUN: %{run} %t_early_image.out

// Mix early and image archive usage
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_early_add.a %t_image_sub.a -o %t_mix.out
// RUN: %{run} %t_mix.out
