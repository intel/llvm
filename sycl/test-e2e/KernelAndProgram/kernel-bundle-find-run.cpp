// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test finds a known kernel and runs it.

#include <sycl/detail/core.hpp>

using namespace sycl;

// Kernel finder
class KernelFinder {
  queue &Queue;
  std::vector<sycl::kernel_id> AllKernelIDs;

public:
  KernelFinder(queue &Q) : Queue(Q) {
    // Obtain kernel bundle
    kernel_bundle Bundle =
        get_kernel_bundle<bundle_state::executable>(Queue.get_context());
    std::cout << "Bundle obtained\n";
    AllKernelIDs = sycl::get_kernel_ids();
    std::cout << "Number of kernels = " << AllKernelIDs.size() << std::endl;
    for (auto K : AllKernelIDs) {
      std::cout << "Kernel obtained: " << K.get_name() << std::endl;
    }
  }

  kernel get_kernel(const char *name) {
    kernel_bundle Bundle =
        get_kernel_bundle<bundle_state::executable>(Queue.get_context());
    for (auto K : AllKernelIDs) {
      auto Kname = K.get_name();
      if (strcmp(name, Kname) == 0) {
        kernel Kernel = Bundle.get_kernel(K);
        std::cout << "Found kernel\n";
        return Kernel;
      }
    }
    std::cout << "No kernel found\n";
    exit(1);
  }
};

void sycl_kernel(queue Queue) {
  range<1> R1{1};
  Queue.submit([&](handler &CGH) {
    CGH.parallel_for<class KernelB>(R1, [=](id<1> WIid) {});
  });
  Queue.wait();
}

int test_sycl_kernel(queue Queue) {
  KernelFinder KF(Queue);

  kernel Kernel = KF.get_kernel("_ZTSZZ11sycl_kernelN4sycl3_V15queueEENKUlRNS0_"
                                "7handlerEE_clES3_E7KernelB");

  range<1> R1{1};
  Queue.submit([&](handler &Handler) { Handler.parallel_for(R1, Kernel); });
  Queue.wait();

  return 0;
}

int main() {
  queue Queue;

  sycl_kernel(Queue);
  std::cout << "sycl_kernel done\n";
  test_sycl_kernel(Queue);
  std::cout << "test_sycl_kernel done\n";

  return 0;
}
