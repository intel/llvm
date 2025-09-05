// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test performs basic check of supporting user defined class that are
// implicitly converted from sycl::item/sycl::nd_item in parallel_for.

#include <iostream>
#include <sycl/detail/core.hpp>

template <int Dimensions> class item_wrapper {
public:
  item_wrapper(sycl::item<Dimensions> it) : m_item(it) {}

  size_t get() { return m_item; }

private:
  sycl::item<Dimensions> m_item;
};

template <int Dimensions> class nd_item_wrapper {
public:
  nd_item_wrapper(sycl::nd_item<Dimensions> it) : m_item(it) {}

  size_t get() { return m_item.get_global_linear_id(); }

private:
  sycl::nd_item<Dimensions> m_item;
};

int main() {
  sycl::queue q;

  // Initialize data array
  const int sz = 16;
  int data[sz] = {0};
  for (int i = 0; i < sz; ++i) {
    data[i] = i;
  }

  // Check user defined sycl::item wrapper
  sycl::buffer<int> data_buf(data, sz);
  q.submit([&](sycl::handler &h) {
    auto buf_acc = data_buf.get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<1>{sz},
                   [=](item_wrapper<1> item) { buf_acc[item.get()] += 1; });
  });
  q.wait();
  bool failed = false;

  {
    sycl::host_accessor buf_acc(data_buf, sycl::read_only);
    for (int i = 0; i < sz; ++i) {
      failed |= (buf_acc[i] != i + 1);
    }
    if (failed) {
      std::cout << "item_wrapper check failed" << std::endl;
      return 1;
    }
  }

  // Check user defined sycl::nd_item wrapper
  q.submit([&](sycl::handler &h) {
    auto buf_acc = data_buf.get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range<1>{sz, 2},
                   [=](nd_item_wrapper<1> item) { buf_acc[item.get()] += 1; });
  });
  q.wait();

  {
    sycl::host_accessor buf_acc(data_buf, sycl::read_only);
    for (int i = 0; i < sz; ++i) {
      failed |= (buf_acc[i] != i + 2);
    }
    if (failed) {
      std::cout << "nd_item_wrapper check failed" << std::endl;
      return 1;
    }
  }

  std::cout << "Test passed" << std::endl;
  return 0;
}
