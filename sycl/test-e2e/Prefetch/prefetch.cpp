// REQUIRES: gpu && (level_zero || opencl)
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;

constexpr size_t N = 128;
constexpr size_t NumWI = 4;
constexpr size_t arrSize = N / NumWI;

#define COMMA ,

#define TEST_PREFETCH_WO_COUNT(PREFETCH_ARG)                                   \
  {                                                                            \
    std::vector<int> res(N);                                                   \
    {                                                                          \
      buffer<int, 1> buf(res.data(), N);                                       \
      q.submit([&](handler &h) {                                               \
        auto acc = buf.get_access<access_mode::write>(h);                      \
        syclex::prefetch(PREFETCH_ARG,                                         \
                         syclex::properties{syclex::prefetch_hint_L1});        \
        h.parallel_for(NumWI, [=](id<1> idx) {                                 \
          for (int i = idx * arrSize; i < idx * arrSize + arrSize; i++)        \
            acc[i] = dataChar[0] * dataChar[0];                                \
        });                                                                    \
      });                                                                      \
      q.wait();                                                                \
    }                                                                          \
    for (int i = 0; i < N; i++)                                                \
      assert(res[i] == dataChar[0] * dataChar[0]);                             \
  }

#define TEST_PREFETCH_W_COUNT(PREFETCH_ARG)                                    \
  {                                                                            \
    std::vector<int> res(N);                                                   \
    {                                                                          \
      buffer<int, 1> buf(res.data(), N);                                       \
      q.submit([&](handler &h) {                                               \
        auto acc = buf.get_access<access_mode::write>(h);                      \
        h.parallel_for(NumWI, [=](id<1> idx) {                                 \
          syclex::prefetch(PREFETCH_ARG, arrSize,                              \
                           syclex::properties{syclex::prefetch_hint_L1});      \
          for (int i = idx * arrSize; i < idx * arrSize + arrSize; i++)        \
            acc[i] = dataChar[i] * dataChar[i];                                \
        });                                                                    \
      });                                                                      \
      q.wait();                                                                \
    }                                                                          \
    for (int i = 0; i < N; i++)                                                \
      assert(res[i] == dataChar[i] * dataChar[i]);                             \
  }

void testPrefetchWithAcc(queue q, const std::vector<int> &data,
                         bool prefetchOneElem = true) {
  std::vector<int> res(N);
  {
    buffer<int, 1> bufRes(res.data(), N);
    buffer<int, 1> bufData(data.data(), N);
    q.submit([&](handler &h) {
      auto accRes = bufRes.get_access<access_mode::write>(h);
      auto accData = bufData.get_access<access_mode::read>(h);
      if (prefetchOneElem)
        syclex::prefetch(accData, id(0),
                         syclex::properties{syclex::prefetch_hint_L1});
      h.parallel_for(NumWI, [=](id<1> idx) {
        if (!prefetchOneElem)
          syclex::prefetch(accData, id(idx * arrSize), arrSize,
                           syclex::properties{syclex::prefetch_hint_L1});
        for (int i = idx * arrSize; i < idx * arrSize + arrSize; i++)
          accRes[i] = prefetchOneElem ? accData[0] * accData[0]
                                      : accData[i] * accData[i];
      });
    });
    q.wait();
  }
  for (int i = 0; i < N; i++)
    assert(res[i] == (prefetchOneElem ? data[0] * data[0] : data[i] * data[i]));
}

int main() {
  queue q;

  if (q.get_device().has(aspect::usm_shared_allocations)) {
    auto *dataChar = malloc_shared<char>(N, q);
    auto *dataVoid = reinterpret_cast<void *>(dataChar);
    auto mPtrChar = address_space_cast<access::address_space::global_space,
                                       access::decorated::yes>(dataChar);
    auto mPtrVoid = address_space_cast<access::address_space::global_space,
                                       access::decorated::yes>(dataVoid);
    std::iota(dataChar, dataChar + N, 0);

    // void prefetch(void* ptr, Properties properties = {});
    TEST_PREFETCH_WO_COUNT(dataVoid)
    // void prefetch(T* ptr, Properties properties = {});
    TEST_PREFETCH_WO_COUNT(dataChar)
    // void prefetch(void* ptr, size_t bytes, Properties properties = {});
    TEST_PREFETCH_W_COUNT(reinterpret_cast<void *>(&dataChar[idx * arrSize]))
    // void prefetch(T* ptr, size_t count, Properties properties = {});
    TEST_PREFETCH_W_COUNT(&dataChar[idx * arrSize])
    // void prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr,
    //               Properties properties = {});
    TEST_PREFETCH_WO_COUNT(mPtrVoid)
    // void prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr,
    //               Properties properties = {});
    TEST_PREFETCH_WO_COUNT(mPtrChar)
    // void prefetch(multi_ptr<void, AddressSpace, IsDecorated> ptr,
    //               size_t bytes, Properties properties = {});
    TEST_PREFETCH_W_COUNT(
        address_space_cast<
            access::address_space::global_space COMMA access::decorated::yes>(
            reinterpret_cast<void *>(&dataChar[idx * arrSize])))
    // void prefetch(multi_ptr<T, AddressSpace, IsDecorated> ptr, size_t count,
    //               Properties properties = {});
    TEST_PREFETCH_W_COUNT(
        address_space_cast<
            access::address_space::global_space COMMA access::decorated::yes>(
            &dataChar[idx * arrSize]))

    free(dataChar, q);
  }
  {
    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);

    // void prefetch(accessor<DataT, Dimensions, AccessMode, target::device,
    //               IsPlaceholder> acc, id<Dimensions> offset,
    //               Properties properties = {});
    testPrefetchWithAcc(q, data);
    // void prefetch(accessor<DataT, Dimensions, AccessMode, target::device,
    //               IsPlaceholder> acc, id<Dimensions> offset, size_t count,
    //               Properties properties = {});
    testPrefetchWithAcc(q, data, false);
  }
}
