// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

queue q;
context ctx = q.get_context();

constexpr size_t SIZE = 128;

template <typename T, typename... Ts>
void test_struct(size_t SIZE, size_t WGSIZE) {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;
  S<T> *buf = malloc_shared<S<T>>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i].val = T(i);
    expected = expected + buf[i].val;
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<S<T>[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       mem[local_id] = buf[local_id];
       group_barrier(it.get_group());
       if (it.get_group().leader()) {
         result = 0;
         for (int i = 0; i < WGSIZE; ++i) {
           result = result + mem[i].val;
         }
         assert(result == expected);
       }
     });
   }).wait();
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_struct<Ts...>(SIZE, WGSIZE);
}

void test_union(size_t SIZE, size_t WGSIZE) {
  U *buf = malloc_shared<U>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  int expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    if (i % 2)
      buf[i].s = S<int>{i};
    else
      buf[i].m = M<int>{i};
    expected = expected + (i % 2) ? buf[i].s.val : buf[i].m.val;
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<U[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental::work_group_memory<int> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       mem[local_id] = buf[local_id];
       group_barrier(it.get_group());
       if (it.get_group().leader()) {
         result = 0;
         for (int i = 0; i < WGSIZE; ++i) {
           result = result + (i % 2) ? mem[i].s.val : mem[i].m.val;
         }
         assert(result == expected);
       }
     });
   }).wait();
  free(buf, q);
}

template <typename T, typename... Ts>
void test(size_t SIZE, size_t WGSIZE, bool UseHelper) {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i);
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<T[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       mem[local_id] = buf[local_id];
       group_barrier(it.get_group());
       if (it.get_group().leader()) {
         result = 0;
         if (!UseHelper) {
           for (int i = 0; i < WGSIZE; ++i) {
             result = result + mem[i];
           }
         } else {
           sum_helper(mem, result, WGSIZE);
         }
         assert(result == expected);
       }
     });
   }).wait();
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test<Ts...>(SIZE, WGSIZE, UseHelper);
}

template <typename T, typename... Ts> void test_marray() {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;
  constexpr size_t WGSIZE = SIZE;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i) / WGSIZE;
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<marray<T, WGSIZE>> mem{cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       constexpr T tolerance = 0.0001;
       // User-defined conversion from work group memory to underlying type is
       // not applied during member access calls so we have to explicitly
       // convert to the value_type ourselves.
       marray<T, WGSIZE> &data = mem;
       data[local_id] = buf[local_id];
       group_barrier(it.get_group());
       if (it.get_group().leader()) {
         result = 0;
         for (int i = 0; i < WGSIZE; ++i) {
           result = result + data[i];
         }
         assert((result - expected) * (result - expected) <= tolerance);
       }
     });
   }).wait();
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_marray<Ts...>();
}

template <typename T, typename... Ts> void test_vec() {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;
  constexpr size_t WGSIZE = 8;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i) / WGSIZE;
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<vec<T, WGSIZE>> mem{cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       constexpr T tolerance = 0.0001;
       vec<T, WGSIZE> &data = mem;
       data[local_id] = buf[local_id];
       group_barrier(it.get_group());
       if (it.get_group().leader()) {
         result = 0;
         for (int i = 0; i < WGSIZE; ++i) {
           result = result + data[i];
         }
         assert((result - expected) * (result - expected) <= tolerance);
       }
     });
   }).wait();
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_vec<Ts...>();
}

template <typename T, typename... Ts> void test_atomic_ref() {
  if (!(sizeof(T) == 4 ||
        (sizeof(T) == 8 && q.get_device().has(aspect::atomic64)))) {
    std::cout << "Invalid type used with atomic_ref!\nSkipping the test!";
    return;
  }
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;
  constexpr size_t WGSIZE = 8;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i);
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<T[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental::work_group_memory<T> result{cgh};
     cgh.parallel_for(ndr, [=](nd_item<> it) {
       size_t local_id = it.get_local_id();
       mem[local_id] = buf[local_id];
       atomic_ref<T, memory_order::acq_rel, memory_scope::work_group>
           atomic_val{result};
       if (it.get_group().leader()) {
         atomic_val.store(0);
       }
       group_barrier(it.get_group());
       atomic_val += mem[local_id];
       group_barrier(it.get_group());
       assert(atomic_val.load() == expected);
     });
   }).wait();
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_atomic_ref<Ts...>();
}

int main() {
  test<int, uint16_t, half, double, float>(SIZE, SIZE /* WorkGroupSize */,
                                           true /* UseHelper */);
  test<int, float, half>(SIZE, SIZE, false);
  test<int, double, char>(SIZE, SIZE / 2, false);
  test<int, bool, char>(SIZE, SIZE / 4, false);
  test<int, float>(SIZE, 1, false);
  test<int, float>(SIZE, 2, true);
  test_marray<float, double, half>();
  test_vec<float, double, half>();
  test_atomic_ref<int, long, float, double>();
  test_struct<int, float, double>(SIZE, 4);
  test_union(SIZE, SIZE);
  test_union(SIZE, SIZE / 2);
  test_union(SIZE, 1);
  test_union(SIZE, 2);
  return 0;
}
