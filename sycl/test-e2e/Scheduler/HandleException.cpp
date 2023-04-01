// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-dead-args-optimization %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Expected exception is generated for OpenCL backend only.
// REQUIRES: opencl
#include <array>
#include <sycl/sycl.hpp>

using namespace sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

constexpr unsigned MAX_WG_SIZE = 4;
constexpr unsigned SIZE = 8;
using ArrayType = std::array<unsigned, SIZE>;

class kernelCompute;

// Return 'true' if an exception was thrown.
bool run_kernel(const unsigned wg_size) {
  ArrayType index;
  const unsigned N = index.size();
  {
    buffer<cl_uint, 1> bufferIdx(index.data(), N);
    queue deviceQueue;
    try {
      deviceQueue.submit([&](handler &cgh) {
        auto accessorIdx = bufferIdx.get_access<sycl_read>(cgh);
        cgh.parallel_for<class kernelCompute>(
            nd_range<1>(range<1>(N), range<1>(wg_size)), [=
        ](nd_item<1> ID) [[cl::reqd_work_group_size(1, 1, MAX_WG_SIZE)]] {
              (void)accessorIdx[ID.get_global_id(0)];
            });
      });
    } catch (nd_range_error &err) {
      return true;
    } catch (...) {
      assert(!"Unknown exception was thrown");
    }
  }
  return false;
}

int main() {
  bool success_exception = run_kernel(MAX_WG_SIZE);
  assert(!success_exception &&
         "Unexpected exception was thrown for success call");
  bool fail_exception = run_kernel(SIZE);
  assert(fail_exception && "No exception was thrown");

  return 0;
}
