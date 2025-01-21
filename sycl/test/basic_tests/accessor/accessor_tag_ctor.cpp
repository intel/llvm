// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only

#include <sycl/sycl.hpp>

constexpr static int size = 1;

void test_accessor_tag_ctor(sycl::buffer<int, size> &buffer) {
  auto read_only_dev_acc = sycl::accessor(buffer, sycl::read_only);
  auto write_only_dev_acc = sycl::accessor(buffer, sycl::write_only);
  auto read_write_dev_acc = sycl::accessor(buffer, sycl::read_write);
  auto read_only_host_acc = sycl::accessor(buffer, sycl::read_only_host_task);
  auto write_only_host_acc = sycl::accessor(buffer, sycl::write_only_host_task);
  auto read_write_host_acc = sycl::accessor(buffer, sycl::read_write_host_task);
  auto read_write_const_acc = sycl::accessor(buffer, sycl::read_constant);
}

void test_accessor_tag_ctor(sycl::handler &cgh,
                            sycl::buffer<int, size> &buffer) {
  auto read_only_dev_acc = sycl::accessor(buffer, cgh, sycl::read_only);
  auto write_only_dev_acc = sycl::accessor(buffer, cgh, sycl::write_only);
  auto read_write_dev_acc = sycl::accessor(buffer, cgh, sycl::read_write);
  auto read_only_host_acc =
      sycl::accessor(buffer, cgh, sycl::read_only_host_task);
  auto write_only_host_acc =
      sycl::accessor(buffer, cgh, sycl::write_only_host_task);
  auto read_write_host_acc =
      sycl::accessor(buffer, cgh, sycl::read_write_host_task);
  auto read_write_const_acc = sycl::accessor(buffer, cgh, sycl::read_constant);
}

void test_accessor_tag_ctor(sycl::buffer<int, size> &buffer,
                            sycl::range<size> access_range) {
  auto read_only_dev_acc =
      sycl::accessor(buffer, access_range, sycl::read_only);
  auto write_only_dev_acc =
      sycl::accessor(buffer, access_range, sycl::write_only);
  auto read_write_dev_acc =
      sycl::accessor(buffer, access_range, sycl::read_write);
  auto read_only_host_acc =
      sycl::accessor(buffer, access_range, sycl::read_only_host_task);
  auto write_only_host_acc =
      sycl::accessor(buffer, access_range, sycl::write_only_host_task);
  auto read_write_host_acc =
      sycl::accessor(buffer, access_range, sycl::read_write_host_task);
  auto read_write_const_acc =
      sycl::accessor(buffer, access_range, sycl::read_constant);
}

void test_accessor_tag_ctor(sycl::handler &cgh, sycl::buffer<int, size> &buffer,
                            sycl::range<size> access_range) {
  auto read_only_dev_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::read_only);
  auto write_only_dev_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::write_only);
  auto read_write_dev_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::read_write);
  auto read_only_host_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::read_only_host_task);
  auto write_only_host_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::write_only_host_task);
  auto read_write_host_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::read_write_host_task);
  auto read_write_const_acc =
      sycl::accessor(buffer, cgh, access_range, sycl::read_constant);
}

void test_accessor_tag_ctor(sycl::buffer<int, size> &buffer,
                            sycl::range<size> access_range,
                            sycl::id<size> access_offset) {
  auto read_only_dev_acc =
      sycl::accessor(buffer, access_range, access_offset, sycl::read_only);
  auto write_only_dev_acc =
      sycl::accessor(buffer, access_range, access_offset, sycl::write_only);
  auto read_write_dev_acc =
      sycl::accessor(buffer, access_range, access_offset, sycl::read_write);
  auto read_only_host_acc = sycl::accessor(buffer, access_range, access_offset,
                                           sycl::read_only_host_task);
  auto write_only_host_acc = sycl::accessor(buffer, access_range, access_offset,
                                            sycl::write_only_host_task);
  auto read_write_host_acc = sycl::accessor(buffer, access_range, access_offset,
                                            sycl::read_write_host_task);
  auto read_write_const_acc =
      sycl::accessor(buffer, access_range, access_offset, sycl::read_constant);
}

void test_accessor_tag_ctor(sycl::handler &cgh, sycl::buffer<int, size> &buffer,
                            sycl::range<size> access_range,
                            sycl::id<size> access_offset) {
  auto read_only_dev_acc =
      sycl::accessor(buffer, cgh, access_range, access_offset, sycl::read_only);
  auto write_only_dev_acc = sycl::accessor(buffer, cgh, access_range,
                                           access_offset, sycl::write_only);
  auto read_write_dev_acc = sycl::accessor(buffer, cgh, access_range,
                                           access_offset, sycl::read_write);
  auto read_only_host_acc = sycl::accessor(
      buffer, cgh, access_range, access_offset, sycl::read_only_host_task);
  auto write_only_host_acc = sycl::accessor(
      buffer, cgh, access_range, access_offset, sycl::write_only_host_task);
  auto read_write_host_acc = sycl::accessor(
      buffer, cgh, access_range, access_offset, sycl::read_write_host_task);
  auto read_write_const_acc = sycl::accessor(
      buffer, cgh, access_range, access_offset, sycl::read_constant);
}
