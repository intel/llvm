// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -w -emit-mlir %s -o -
// XFAIL: *

// Possibly fails due to the bad handling of `memref<?x!llvm.array<NxTy>>` in
// `VisitArraySubscriptExpr`

#include <sycl/sycl.hpp>

using namespace sycl;

int main(){
  queue q;
  range range{8};

  buffer<float, 1> buf{nullptr, range};

  q.submit([&](handler& cgh){
    auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
    const float array[] = {1, 2, 3, 4, 5, 6, 7, 8};
    cgh.parallel_for(range, [=](id<1> i){ acc[i] = array[static_cast<unsigned>(i)]; });
  });

  return 0;
}
