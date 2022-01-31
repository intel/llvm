// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

// This test performs basic check of supporting user defined class that are
// implicitly converted from sycl::item/sycl::nd_item in parallel_for.

#include <CL/sycl.hpp>
#include <iostream>

template <int Dimensions> class item_wrapper {
public:
  item_wrapper(sycl::item<Dimensions> it) : m_item(it) {}

private:
  sycl::item<Dimensions> m_item;
};

template <int Dimensions> class nd_item_wrapper {
public:
  nd_item_wrapper(sycl::nd_item<Dimensions> it) : m_item(it) {}

private:
  sycl::nd_item<Dimensions> m_item;
};

template <int Dimensions, typename T> class item_wrapper2 {
public:
  item_wrapper2(sycl::item<Dimensions> it) : m_item(it), m_value(T()) {}

private:
  sycl::item<Dimensions> m_item;
  T m_value;
};

template <int Dimensions, typename T> class nd_item_wrapper2 {
public:
  nd_item_wrapper2(sycl::nd_item<Dimensions> it) : m_item(it), m_value(T()) {}

private:
  sycl::nd_item<Dimensions> m_item;
  T m_value;
};

int main() {
  sycl::queue q;

  q.parallel_for(sycl::range<1>{1}, [=](item_wrapper<1> item) {});
  q.parallel_for(sycl::nd_range<1>{1, 1}, [=](nd_item_wrapper<1> item) {});
  q.parallel_for(sycl::range<1>{1}, [=](item_wrapper2<1, int> item) {});
  q.parallel_for(sycl::nd_range<1>{1, 1},
                 [=](nd_item_wrapper2<1, int> item) {});

  return 0;
}
