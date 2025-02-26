// Test compiler behaviors for -fintelfpga combining -fsycl-link=image and
// -fsycl-link=early.

// REQUIRES: opencl-aot, accelerator

// Remove any archives
// RUN: rm -f %t_*.a

// Build main
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp -c -o %t_main.o

// Build any early archive binaries.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=early %S/Inputs/fpga_sub.cpp   -o %t_early_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=early %S/Inputs/fpga_sub_x.cpp -o %t_early_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=early %S/Inputs/fpga_add_x.cpp -o %t_early_add_x.a

// Build any image archive binaries.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_sub.cpp   -o %t_image_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_sub_x.cpp -o %t_image_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_add_x.cpp -o %t_image_add_x.a

// Build using various combinations of archives and source.
// RUN: %clangxx -fintelfpga -fsycl %t_main.o %S/Inputs/fpga_add.cpp %t_image_sub.a %t_early_add_x.a %t_image_sub_x.a -o %t_mix.out
// RUN: %{run} %t_mix.out
