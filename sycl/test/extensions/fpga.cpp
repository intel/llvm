// RUN: %clangxx -fsycl %s -o %t.out

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
namespace intelfpga {
template <unsigned ID> struct ethernet_pipe_id {
  static constexpr unsigned id = ID;
};

using ethernet_read_pipe =
    sycl::INTEL::kernel_readable_io_pipe<ethernet_pipe_id<0>, int, 0>;
using ethernet_write_pipe =
    sycl::INTEL::kernel_writeable_io_pipe<ethernet_pipe_id<1>, int, 0>;
} // namespace intelfpga

int main() {
  sycl::queue Queue;
  /* Check buffer_location property  */
  sycl::buffer<int, 1> Buf{sycl::range{1}};
  Queue.submit([&](sycl::handler &CGH) {
    sycl::ONEAPI::accessor_property_list PL{sycl::INTEL::buffer_location<1>};
    sycl::accessor Acc(Buf, CGH, sycl::write_only, PL);
    CGH.single_task<class Test>([=]() { Acc[0] = 42; });
  });
  Queue.wait();

  auto Acc = Buf.template get_access<sycl::access::mode::read_write>();
  assert(Acc[0] == 42 && "Value mismatch");

  /*Check FPGA-related device parameters*/
  if (!Queue.get_device()
           .get_info<cl::sycl::info::device::kernel_kernel_pipe_support>()) {
    std::cout << "SYCL_INTEL_data_flow_pipes not supported, skipping"
              << std::endl;
    return 0;
  }

  /*Check pipes interfaces*/
  Queue.submit([&](cl::sycl::handler &cgh) {
    auto write_acc = Buf.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.single_task<class bl_io_transfer>([=]() {
      write_acc[0] = intelfpga::ethernet_read_pipe::read();
      intelfpga::ethernet_write_pipe::write(write_acc[0]);
    });
  });

  using Pipe = cl::sycl::INTEL::pipe<class PipeName, int>;
  cl::sycl::buffer<int, 1> readBuf(1);
  Queue.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<class writer>([=]() {
      bool SuccessCode = false;
      do {
        Pipe::write(42, SuccessCode);
      } while (!SuccessCode);
    });
  });

  /*Check LSU interface*/
  {
    cl::sycl::buffer<int, 1> output_buffer(1);
    auto *in_ptr = cl::sycl::malloc_host<int>(1, Queue.get_context());

    Queue.submit([&](cl::sycl::handler &cgh) {
      auto output_accessor =
          output_buffer.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.single_task<class kernel>([=] {
        cl::sycl::host_ptr<int> input_ptr(in_ptr);
        auto output_ptr = output_accessor.get_pointer();

        using PrefetchingLSU =
            cl::sycl::INTEL::lsu<cl::sycl::INTEL::prefetch<true>,
                                 cl::sycl::INTEL::statically_coalesce<false>>;

        using BurstCoalescedLSU =
            cl::sycl::INTEL::lsu<cl::sycl::INTEL::burst_coalesce<true>,
                                 cl::sycl::INTEL::statically_coalesce<false>>;

        using CachingLSU =
            cl::sycl::INTEL::lsu<cl::sycl::INTEL::burst_coalesce<true>,
                                 cl::sycl::INTEL::cache<1024>,
                                 cl::sycl::INTEL::statically_coalesce<false>>;

        using PipelinedLSU = cl::sycl::INTEL::lsu<>;

        int X = PrefetchingLSU::load(input_ptr); // int X = input_ptr[0]
        int Y = CachingLSU::load(input_ptr + 1); // int Y = input_ptr[1]

        BurstCoalescedLSU::store(output_ptr, X); // output_ptr[0] = X
        PipelinedLSU::store(output_ptr + 1, Y);  // output_ptr[1] = Y
      });
    });
  }

  return 0;
}
