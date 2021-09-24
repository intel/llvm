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

  // explicitly instantiate a few more class pointers to check if there are some
  // issues with them:

  buffer<vec<int, 3>, 2> *p1;
  (void) p1;
  accessor<int, 1> *p2;
  (void) p2;
  accessor<vec<float, 2>, 2, access_mode::read> *p3;
  (void) p3;
  marray<double, 7> *p4;
  (void) p4;
  marray<short, 3> *p5;
  (void) p5;
  kernel_bundle<bundle_state::input> *p6;
  (void) p6;
  kernel_bundle<bundle_state::object> *p7;
  (void) p7;
  kernel_bundle<bundle_state::executable> *p8;
  (void) p8;
  device_image<bundle_state::input> *p9;
  (void) p9;
  device_image<bundle_state::object> *p10;
  (void) p10;
  device_image<bundle_state::executable> *p11;
  (void) p11;
  return 0;
}
