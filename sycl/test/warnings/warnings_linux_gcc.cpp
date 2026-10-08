// REQUIRES: linux
// RUN: gcc -Wall -Wextra -Werror -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING -I%sycl_include %s -L%sycl_libs_dir %sycl_lib -lstdc++ -o %t

#include "./warnings.cpp"
