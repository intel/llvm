#include <sycl/ext/intel/fpga_extensions.hpp>

namespace intelfpga {
template <unsigned ID> struct ethernet_pipe_id {
  static constexpr unsigned id = ID;
};

using ethernet_read_pipe =
    sycl::ext::intel::kernel_readable_io_pipe<ethernet_pipe_id<0>, int, 0>;
using ethernet_write_pipe =
    sycl::ext::intel::kernel_writeable_io_pipe<ethernet_pipe_id<1>, int, 0>;
} // namespace intelfpga
