// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==----- dependent_event_async_exception.cpp - Test for event async exceptions
//-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>
#include <unordered_set>

struct test_exception {
  std::string name;
};

class test_exception_handler {
public:
  test_exception_handler()
      : queue{[this](sycl::exception_list el) { capture(std::move(el)); }} {}
  sycl::queue &get_queue() { return queue; }

  bool has(const std::string &name) const {
    return captured_exceptions.count(name) != 0;
  }

  size_t count() const { return captured_exceptions.size(); }

  void clear() { captured_exceptions.clear(); }

private:
  std::unordered_set<std::string> captured_exceptions;
  sycl::queue queue;

  void capture(sycl::exception_list el) {
    for (auto &e : el) {
      try {
        std::rethrow_exception(e);
      } catch (test_exception &te) {
        captured_exceptions.insert(te.name);
      }
    }
  }
};

static sycl::event
make_throwing_host_event(sycl::queue &queue, std::string name,
                         const std::vector<sycl::event> &dependencies = {}) {
  return queue.submit([name, &dependencies](sycl::handler &cgh) {
    for (auto &dep : dependencies) {
      cgh.depends_on(dep);
    }
    cgh.host_task([name](auto) { throw test_exception{name}; });
  });
}

int main() {
  {
    test_exception_handler teh1;
    test_exception_handler teh2;

    auto e1 = make_throwing_host_event(teh1.get_queue(), "some-error");
    auto e2 = make_throwing_host_event(teh2.get_queue(), "another-error", {e1});

    e2.wait_and_throw();

    assert(teh2.count() == 1);
    assert(teh2.has("another-error"));

    assert(teh1.count() == 1);
    assert(teh1.has("some-error"));
  }
  {
    int data = 0;
    {
      sycl::buffer<int, 1> Buf(&data, sycl::range<1>(1));
      test_exception_handler teh1;
      test_exception_handler teh2;

      auto e1 = teh1.get_queue().submit([&](sycl::handler &cgh) {
        auto B = sycl::accessor(Buf, cgh, sycl::read_write_host_task);
        cgh.host_task([=]() {
          B[0] = 10;
          throw test_exception{"some-error"};
        });
      });

      auto e2 = teh2.get_queue().submit([&](sycl::handler &cgh) {
        auto B = sycl::accessor(Buf, cgh, sycl::read_write_host_task);
        cgh.host_task([=]() {
          B[0] *= 10;
          throw test_exception{"another-error"};
        });
      });

      e2.wait_and_throw();

      assert(data == 100);
      assert(teh2.count() == 1);
      assert(teh2.has("another-error"));

      assert(teh1.count() == 1);
      assert(teh1.has("some-error"));
    }
  }
  {
    int data1 = 0, data2 = 0;
    {
      sycl::buffer<int, 1> Buf1(&data1, sycl::range<1>(1));
      sycl::buffer<int, 1> Buf2(&data2, sycl::range<1>(1));
      test_exception_handler teh;

      auto e1 = teh.get_queue().submit([&](sycl::handler &cgh) {
        auto B = sycl::accessor(Buf1, cgh, sycl::read_write_host_task);
        cgh.host_task([=]() {
          B[0] = 10;
          throw test_exception{"some-error"};
        });
      });

      auto e2 = teh.get_queue().submit([&](sycl::handler &cgh) {
        auto B = sycl::accessor(Buf2, cgh, sycl::read_write_host_task);
        cgh.host_task([=]() {
          B[0] = 20;
          throw test_exception{"another-error"};
        });
      });

      e2.wait_and_throw();

      assert(data1 == 10);
      assert(data2 == 20);
      assert(teh.count() == 2);
      assert(teh.has("another-error"));
      assert(teh.has("some-error"));
    }
  }
}
