// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

int data[] = {0, 1, 2, 3};

int main() {
  namespace syclex = sycl::ext::oneapi::experimental;
  void *dataPtrVoid = data;
  int *dataPtrInt = data;
  auto prop = syclex::properties{syclex::prefetch_hint_L1};

  {
    sycl::buffer<int, 1> buf(data, 4);
    sycl::queue q;
    q.submit([&](sycl::handler &h) {
      auto acc = buf.get_access<sycl::access_mode::read>(h);
      h.parallel_for<class Kernel>(
          sycl::nd_range<1>(1, 1), ([=](sycl::nd_item<1> index) {
            syclex::prefetch(dataPtrVoid, prop);
            syclex::prefetch(dataPtrVoid, 16, prop);

            syclex::prefetch(dataPtrInt, prop);
            syclex::prefetch(dataPtrInt, 4, prop);

            auto mPtrVoid = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(dataPtrVoid);
            syclex::prefetch(mPtrVoid, prop);
            syclex::prefetch(mPtrVoid, 16, prop);

            auto mPtrInt = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(dataPtrInt);
            syclex::prefetch(mPtrInt, prop);
            syclex::prefetch(mPtrInt, 8, prop);

            syclex::prefetch(acc, sycl::id(0), prop);
            syclex::prefetch(acc, sycl::id(0), 4, prop);

            auto g = index.get_group();
            syclex::joint_prefetch(g, dataPtrVoid, prop);
            syclex::joint_prefetch(g, dataPtrVoid, 16, prop);

            syclex::joint_prefetch(g, dataPtrInt, prop);
            syclex::joint_prefetch(g, dataPtrInt, 4, prop);

            syclex::joint_prefetch(g, mPtrVoid, prop);
            syclex::joint_prefetch(g, mPtrVoid, 16, prop);

            syclex::joint_prefetch(g, mPtrInt, prop);
            syclex::joint_prefetch(g, mPtrInt, 8, prop);

            syclex::joint_prefetch(g, acc, sycl::id(0), prop);
            syclex::joint_prefetch(g, acc, sycl::id(0), 4, prop);
          }));
    });
    q.wait();
  }

  return 0;
}
