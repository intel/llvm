// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <CL/sycl/detail/circular_buffer.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

// This test contains basic checks for cl::sycl::detail::CircularBuffer
void checkEquality(const cl::sycl::detail::CircularBuffer<int> &CB,
                   const std::vector<int> &V) {
  assert(std::equal(CB.begin(), CB.end(), V.begin()));
}

int main() {
  const size_t Capacity = 6;
  cl::sycl::detail::CircularBuffer<int> CB{Capacity};
  assert(CB.capacity() == Capacity);
  assert(CB.empty());

  int nextValue = 0;
  for (; nextValue < Capacity; ++nextValue) {
    assert(CB.size() == nextValue);
    CB.push_back(nextValue);
  }
  assert(CB.full() && CB.size() == CB.capacity());
  checkEquality(CB, {0, 1, 2, 3, 4, 5});

  CB.push_back(nextValue++);
  checkEquality(CB, {1, 2, 3, 4, 5, 6});
  CB.push_front(nextValue++);
  checkEquality(CB, {7, 1, 2, 3, 4, 5});

  assert(CB.front() == 7);
  assert(CB.back() == 5);

  CB.erase(CB.begin() + 2);
  checkEquality(CB, {7, 1, 3, 4, 5});
  CB.erase(CB.begin(), CB.begin() + 2);
  checkEquality(CB, {3, 4, 5});

  CB.pop_back();
  checkEquality(CB, {3, 4});
  CB.pop_front();
  checkEquality(CB, {4});
}
