// UNSUPPORTED: cuda
// CUDA compilation and runtime do not yet support sub-groups.
//
// RUN: %clangxx -fsycl -std=c++14 %s -o %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -std=c++14 -D SG_GPU %s -o %t_gpu.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t_gpu.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "helper.hpp"
#include <CL/sycl.hpp>
#include <complex>

template <typename T, class BinaryOperation>
class generic_reduce;

using namespace cl::sycl;

template <typename T, class BinaryOperation>
void check_op(queue &Queue, T init, BinaryOperation op, bool skip_init = false,
              size_t G = 240, size_t L = 60) {
  try {
    nd_range<1> NdRange(G, L);
    buffer<T> buf(G);
    buffer<size_t> sgsizebuf(1);
    Queue.submit([&](handler &cgh) {
      auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>(cgh);
      auto acc = buf.template get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<generic_reduce<T, BinaryOperation>>(
          NdRange, [=](nd_item<1> NdItem) {
            intel::sub_group sg = NdItem.get_sub_group();
            if (skip_init) {
              acc[NdItem.get_global_id(0)] =
                  reduce(sg, T(NdItem.get_global_id(0)), op);
            } else {
              acc[NdItem.get_global_id(0)] =
                  reduce(sg, T(NdItem.get_global_id(0)), init, op);
            }
            if (NdItem.get_global_id(0) == 0)
              sgsizeacc[0] = sg.get_max_local_range()[0];
          });
    });
    auto acc = buf.template get_access<access::mode::read_write>();
    auto sgsizeacc = sgsizebuf.get_access<access::mode::read_write>();
    size_t sg_size = sgsizeacc[0];
    int WGid = -1, SGid = 0;
    T result = init;
    for (int j = 0; j < G; j++) {
      if (j % L % sg_size == 0) {
        SGid++;
        result = init;
        for (int i = j; (i % L && i % L % sg_size) || (i == j); i++) {
          result = op(result, T(i));
        }
      }
      if (j % L == 0) {
        WGid++;
        SGid = 0;
      }
      std::string name =
          std::string("reduce_") + typeid(BinaryOperation).name();
      exit_if_not_equal(acc[j], result, name.c_str());
    }
  } catch (exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    exit(1);
  }
}

int main() {
  queue Queue;
  if (!core_sg_supported(Queue.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  size_t G = 240;
  size_t L = 60;

  // Test user-defined type
  // Use complex as a proxy for this
  using UDT = std::complex<float>;
  check_op<UDT>(Queue, UDT(L, L), intel::plus<UDT>(), false, G, L);
  check_op<UDT>(Queue, UDT(0, 0), intel::plus<UDT>(), true, G, L);

  // Test user-defined operator
  auto UDOp = [=](const auto& lhs, const auto& rhs) { return lhs + rhs; };
  check_op<int>(Queue, int(L), UDOp, false, G, L);
  check_op<int>(Queue, int(0), UDOp, true, G, L);

  // Test both user-defined type and operator
  check_op<UDT>(Queue, UDT(L, L), UDOp, false, G, L);
  check_op<UDT>(Queue, UDT(0, 0), UDOp, true, G, L);

  std::cout << "Test passed." << std::endl;
  return 0;
}
