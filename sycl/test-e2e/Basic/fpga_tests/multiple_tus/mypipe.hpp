#include <cstdint>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#pragma once

struct KernelFunctor {
  using my_pipe =
      sycl::ext::intel::experimental::pipe<struct sample_host_pipe, uint32_t>;
  SYCL_EXTERNAL void operator()() const;
};
