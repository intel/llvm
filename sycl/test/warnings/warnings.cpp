// RUN: %clangxx -fsycl --no-system-header-prefix=CL/sycl -fsyntax-only -Wall -Wextra -Werror -Wno-ignored-attributes -Wno-deprecated-declarations -Wpessimizing-move -Wunused-variable -Wmismatched-tags -Wunneeded-internal-declaration -Wno-unknown-cuda-version -Wno-unused-parameter -Wno-unused-command-line-argument %s
// RUN: %clangxx -fsycl -E --no-system-header-prefix=CL/sycl %s -o %t.ii
// RUN: %clangxx -fsycl -fsyntax-only -Wall -Wextra -Werror -Wno-ignored-attributes -Wno-deprecated-declarations -Wpessimizing-move -Wunused-variable -Wmismatched-tags -Wunneeded-internal-declaration -Wno-unknown-cuda-version -Wno-unused-parameter -Wno-unused-command-line-argument %t.ii
#include <CL/sycl.hpp>

using namespace cl::sycl;
int main() {
  vec<long, 4> newVec;
  queue myQueue;
  buffer<vec<long, 4>, 1> resultBuf{&newVec, range<1>{1}};
  auto event = myQueue.submit([&](handler &cgh) {
    auto writeResult = resultBuf.get_access<access::mode::write>(cgh);
    cgh.single_task<class kernel_name>([=]() {
      writeResult[0] = (vec<int, 4>{1, 2, 3, 4}).template convert<long>();
    });
  });
  (void)event;
  return 0;
}

// explicitly instantiate a few more classes to check if there are some issues
// with them:

namespace sycl {

template class buffer<vec<int, 3>, 2>;

template class accessor<int, 1>;
template class accessor<vec<float, 2>, 2, access_mode::read>;

template class marray<double, 7>;
template class marray<short, 3>;

template class kernel_bundle<bundle_state::input>;
template class kernel_bundle<bundle_state::object>;
template class kernel_bundle<bundle_state::executable>;

template class device_image<bundle_state::input>;
template class device_image<bundle_state::object>;
template class device_image<bundle_state::executable>;

}
