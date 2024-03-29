// Test compiler behaviors for -fintelfpga with -fsycl-link=image.

// REQUIRES: opencl-aot, accelerator

// Remove any archives
// RUN: rm -f %t_*.a

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Build any image archive binaries.
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_sub.cpp   -o %t_image_sub.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_add.cpp   -o %t_image_add.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_sub_x.cpp -o %t_image_sub_x.a
// RUN: %clangxx -fintelfpga -fsycl -fsycl-link=image %S/Inputs/fpga_add_x.cpp -o %t_image_add_x.a
////////////////////////////////////////////////////////////////////////////////
// Use a variety of archive orders
////////////////////////////////////////////////////////////////////////////////
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_image_add.a %t_image_sub.a %t_image_add_x.a %t_image_sub_x.a -o %t_image.out
// RUN: %{run} %t_image.out
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_image_sub_x.a %t_image_add.a %t_image_sub.a %t_image_add_x.a -o %t_image.out
// RUN: %{run} %t_image.out
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_image_add_x.a %t_image_sub_x.a %t_image_add.a %t_image_sub.a -o %t_image.out
// RUN: %{run} %t_image.out
// RUN: %clangxx -fintelfpga -fsycl %S/Inputs/fpga_main.cpp %t_image_sub.a %t_image_add_x.a %t_image_sub_x.a %t_image_add.a -o %t_image.out
// RUN: %{run} %t_image.out
