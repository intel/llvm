#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

class AccessorIteratorTest : public ::testing::Test {
public:
  AccessorIteratorTest() {}

  template <int Dimensions, typename T = int>
  void checkFullCopyThroughIterator(const sycl::range<Dimensions> &shape) {
    std::vector<T> reference(shape.size());
    std::iota(reference.begin(), reference.end(), 0);
    sycl::buffer<T, Dimensions> buffer(reference.data(), shape);
    auto accessor = buffer.template get_access<sycl::access_mode::read_write>();
    std::vector<T> copied;
    auto I = accessor.begin();
    I = accessor.end();
    for (auto i = accessor.begin(), e = accessor.end(); i != e; ++i) {
      copied.push_back(*i);
    }

    ASSERT_EQ(copied.size(), reference.size());
    for (size_t i = 0, e = reference.size(); i < e; ++i) {
      ASSERT_EQ(copied[i], reference[i]);
    }
  }

  template <int TotalDimensions, int CurrentDimension = 3, typename Container,
            typename... Indices>
  auto &&accessHelper(Container &&C, int Idx, Indices... Ids) {
    if constexpr (CurrentDimension > TotalDimensions) {
      (void)Idx;
      return accessHelper<TotalDimensions, CurrentDimension - 1>(C, Ids...);
    } else
      return accessHelper<TotalDimensions, CurrentDimension - 1>(C[Idx],
                                                                 Ids...);
  }

  template <int TotalDimensions, int CurrentDimension = 3, typename Container>
  auto &&accessHelper(Container &&C, int Idx) {
    return C[Idx];
  }

  template <int Dimensions, typename T = int>
  void checkPartialCopyThroughIteratorWithoutOffset(
      const sycl::range<Dimensions> &fullShape,
      const sycl::range<Dimensions> &copyShape) {
    std::vector<T> reference(fullShape.size());
    std::iota(reference.begin(), reference.end(), 0);
    sycl::buffer<T, Dimensions> buffer(reference.data(), fullShape);
    std::vector<T> copied;
    {
      auto accessor =
          buffer.template get_access<sycl::access_mode::read_write>(copyShape);
      auto I = accessor.begin();
      I = accessor.end();
      for (auto i = accessor.begin(), e = accessor.end(); i != e; ++i) {
        copied.push_back(*i);
      }
    }

    ASSERT_EQ(copied.size(), copyShape.size());

    {
      auto fullAccessor = buffer.template get_access<sycl::access_mode::read>();
      size_t linearId = 0;
      sycl::id<3> shapeToCheck(Dimensions > 2 ? copyShape[Dimensions - 3] : 1,
                               Dimensions > 1 ? copyShape[Dimensions - 2] : 1,
                               copyShape[Dimensions - 1]);
      for (size_t z = 0; z < shapeToCheck[0]; ++z) {
        for (size_t y = 0; y < shapeToCheck[1]; ++y) {
          for (size_t x = 0; x < shapeToCheck[2]; ++x) {
            auto value = accessHelper<Dimensions>(fullAccessor, z, y, x);
            ASSERT_EQ(copied[linearId], value);
            ++linearId;
          }
        }
      }
    }
  }
};

TEST_F(AccessorIteratorTest, ImplementationDetails) {
  std::vector<int> reference(5);
  std::iota(reference.begin(), reference.end(), 0);
  sycl::buffer<int> buffer(reference.data(), sycl::range<1>{reference.size()});
  auto accessor = buffer.template get_access<sycl::access_mode::read_write>();
  {
    auto It = accessor.begin();
    // Check that It can't be decremented past begin
    ASSERT_EQ(--It, accessor.begin());
    ASSERT_EQ(It - 1, accessor.begin());
    ASSERT_EQ(It -= 1, accessor.begin());
    ASSERT_EQ(It - 10, accessor.begin());
    ASSERT_EQ(It -= 10, accessor.begin());
  }
  {
    auto It = accessor.end();
    // Check that It can't be incremented past end
    ASSERT_EQ(++It, accessor.end());
    ASSERT_EQ(It + 1, accessor.end());
    ASSERT_EQ(It += 1, accessor.end());
    ASSERT_EQ(It + 10, accessor.end());
    ASSERT_EQ(It += 10, accessor.end());
  }
}

// FIXME: consider turning this into parameterized test to check various
// accessor types
TEST_F(AccessorIteratorTest, IteratorTraits) {
  using IteratorT = sycl::accessor<int>::iterator;
  ASSERT_TRUE(
      (std::is_same_v<sycl::accessor<int>::difference_type,
                      std::iterator_traits<IteratorT>::difference_type>));
  ASSERT_TRUE((std::is_same_v<sycl::accessor<int>::value_type,
                              std::iterator_traits<IteratorT>::value_type>));
  ASSERT_TRUE((std::is_same_v<sycl::accessor<int>::value_type *,
                              std::iterator_traits<IteratorT>::pointer>));
  ASSERT_TRUE((std::is_same_v<sycl::accessor<int>::reference,
                              std::iterator_traits<IteratorT>::reference>));
  ASSERT_TRUE(
      (std::is_same_v<std::random_access_iterator_tag,
                      std::iterator_traits<IteratorT>::iterator_category>));
}

// Based on requirements listed at
// https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
TEST_F(AccessorIteratorTest, LegacyRandomAccessIteratorRequirements) {
  using IteratorT = sycl::accessor<int>::iterator;
  IteratorT It;
  auto &RefToIt = It;
  ASSERT_TRUE((std::is_same_v<IteratorT &, decltype(RefToIt += 3)>));
  ASSERT_TRUE((std::is_same_v<IteratorT, decltype(It + 3)>));
  ASSERT_TRUE((std::is_same_v<IteratorT, decltype(3 + It)>));
  ASSERT_TRUE((std::is_same_v<IteratorT &, decltype(RefToIt -= 3)>));
  ASSERT_TRUE((std::is_same_v<IteratorT, decltype(It - 3)>));
  IteratorT It2;
  ASSERT_TRUE((std::is_same_v<std::iterator_traits<IteratorT>::difference_type,
                              decltype(It - It2)>));
  ASSERT_TRUE(
      (std::is_convertible_v<decltype(It[3]),
                             std::iterator_traits<IteratorT>::reference>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It < It2), bool>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It <= It2), bool>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It > It2), bool>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It >= It2), bool>));
}

// Based on notes listed at
// https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
TEST_F(AccessorIteratorTest, LegacyRandomAccessIteratorRequirementsExtra) {
  std::vector<int> reference(6);
  std::iota(reference.begin(), reference.end(), 0);
  sycl::buffer<int> buffer(reference.data(), sycl::range<1>{reference.size()});
  auto accessor = buffer.template get_access<sycl::access_mode::read_write>();
  auto It = accessor.begin();
  It += 3;

  { // It += n should be equivalent to incrementint/decrementing It n times
    for (int n = -3; n <= 3; ++n) {
      auto It1 = It;
      auto It2 = It;
      It1 += n;

      if (n < 0) {
        int i = n;
        while (i++)
          --It2;
      } else {
        int i = n;
        while (i--)
          ++It2;
      }

      ASSERT_EQ(It1, It2);
    }
  }

  { // It + n == n + It
    for (int n = -3; n <= 3; ++n) {
      ASSERT_EQ(It + n, n + It);
    }
  }

  {
    auto It1 = accessor.begin();
    auto It2 = accessor.end();
    ASSERT_EQ(It - It1, It1 - It);
    ASSERT_EQ(It - It2, It2 - It);
    ASSERT_EQ(It2, It + (It2 - It));
    ASSERT_EQ(It, It1 + (It - It1));
  }

  {
    auto It1 = accessor.begin();
    auto It2 = accessor.begin();
    auto It3 = accessor.end();

    ASSERT_TRUE(!(It1 < It2));
    ASSERT_TRUE(It1 < It); // precondition for the next check
    ASSERT_TRUE(!(It < It1));
    ASSERT_TRUE(It < It3); // precondition for the next check
    ASSERT_TRUE(It1 < It3);

    ASSERT_FALSE(It3 < It);
    ASSERT_FALSE(It == It3);
  }
}

// Based on requirements listed at
// https://en.cppreference.com/w/cpp/named_req/BidirectionalIterator
TEST_F(AccessorIteratorTest, LegacyBidirectionalIteratorRequirements) {
  using IteratorT = sycl::accessor<int>::iterator;
  IteratorT It;
  ASSERT_TRUE((std::is_same_v<IteratorT &, decltype(--It)>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It--), const IteratorT &>));
  ASSERT_TRUE((std::is_same_v<std::iterator_traits<IteratorT>::reference,
                              decltype(*It--)>));
}

// Based on requirements listed at
// https://en.cppreference.com/w/cpp/named_req/ForwardIterator
TEST_F(AccessorIteratorTest, LegacyForwardIteratorRequirements) {
  using IteratorT = sycl::accessor<int>::iterator;
  ASSERT_TRUE(std::is_default_constructible_v<IteratorT>);
  IteratorT It;
  ASSERT_TRUE((std::is_same_v<IteratorT, decltype(It++)>));
  ASSERT_TRUE((std::is_same_v<std::iterator_traits<IteratorT>::reference,
                              decltype(*It++)>));
  IteratorT It2;
  ASSERT_TRUE((std::is_convertible_v<decltype(It == It2), bool>));
  ASSERT_TRUE((std::is_convertible_v<decltype(It != It2), bool>));
}

TEST_F(AccessorIteratorTest, MultipassGuarantee) {
  std::vector<int> reference(5);
  std::iota(reference.begin(), reference.end(), 0);
  sycl::buffer<int> buffer(reference.data(), sycl::range<1>{reference.size()});
  auto accessor = buffer.template get_access<sycl::access_mode::read_write>();
  auto It1 = accessor.begin();
  auto It2 = accessor.begin();

  while (It1 != accessor.end()) {
    ASSERT_EQ(It1, It2);
    ASSERT_EQ(*It1, *It2);
    ASSERT_EQ(++It1, ++It2);
  }

  It1 = accessor.begin();
  It2 = It1;
  ASSERT_EQ(((void)++It2, *It1), *It1);
}

// Based on requirements listead at
// https://en.cppreference.com/w/cpp/named_req/Iterator
TEST_F(AccessorIteratorTest, LegacyIteratorRequirements) {
  using IteratorT = sycl::accessor<int>::iterator;
  ASSERT_TRUE(std::is_copy_constructible_v<IteratorT>);
  ASSERT_TRUE(std::is_copy_assignable_v<IteratorT>);
  ASSERT_TRUE(std::is_destructible_v<IteratorT>);
  ASSERT_TRUE(std::is_swappable_v<IteratorT>);
  IteratorT It;
  ASSERT_TRUE((std::is_same_v<IteratorT &, decltype(++It)>));
  ASSERT_TRUE((std::is_same_v<std::iterator_traits<IteratorT>::reference,
                              decltype(*It)>));
}

TEST_F(AccessorIteratorTest, FullCopy1D) {
  ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIterator(sycl::range<1>{10}));
}

TEST_F(AccessorIteratorTest, FullCopy2D) {
  ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIterator(sycl::range<2>{2, 5}));
  ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIterator(sycl::range<2>{5, 2}));
  ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIterator(sycl::range<2>{1, 10}));
  ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIterator(sycl::range<2>{10, 1}));
}

TEST_F(AccessorIteratorTest, FullCopy3D) {
  ASSERT_NO_FATAL_FAILURE(
      checkFullCopyThroughIterator(sycl::range<3>{3, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(
      checkFullCopyThroughIterator(sycl::range<3>{1, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(
      checkFullCopyThroughIterator(sycl::range<3>{3, 1, 3}));
  ASSERT_NO_FATAL_FAILURE(
      checkFullCopyThroughIterator(sycl::range<3>{3, 3, 1}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithoutOffset1D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<1>{10}, sycl::range<1>{5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<1>{10}, sycl::range<1>{10}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithoutOffset2D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<2>{5, 5}, sycl::range<2>{5, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<2>{5, 5}, sycl::range<2>{2, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<2>{5, 5}, sycl::range<2>{5, 2}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 2}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithoutOffset3D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{5, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 5, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{5, 5, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIteratorWithoutOffset(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{1, 2, 3}));
}
