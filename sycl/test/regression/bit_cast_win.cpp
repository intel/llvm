// DEFINE: %{md_flag} = %if debug_sycl_library %{/MD%} %else %{/MDd%}
// RUN: %clangxx %fsycl -fsycl-host-compiler=cl -fsycl-host-compiler-options='/std:c++17 /Zc:__cplusplus %{md_flag}' %s -o %t.out
// UNSUPPORTED: linux

#include "bit_cast.hpp"
