#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

class AccessorIteratorTest : public ::testing::Test {
public:
  template <int Dimensions, typename T = int>
  void checkWriteThroughIterator(const sycl::range<Dimensions> &fullShape,
                                 const sycl::range<Dimensions> &fillShape,
                                 const sycl::id<Dimensions> &offset) {
    std::vector<T> data(fullShape.size(), T{});
    sycl::buffer buffer(data.data(), fullShape);
    {
      auto accessor = buffer.template get_access<sycl::access_mode::write>(
          fillShape, offset);
      T linear_id = 1;
      for (auto it = accessor.begin(), e = accessor.end(); it != e; ++it) {
        *it = linear_id;
        linear_id += 1;
      }
    }

    sycl::id<3> offsetToUse(Dimensions > 2 ? offset[Dimensions - 3] : 0,
                            Dimensions > 1 ? offset[Dimensions - 2] : 0,
                            offset[Dimensions - 1]);

    sycl::id<3> shapeToCheck(
        (Dimensions > 2 ? fillShape[Dimensions - 3] : 1) + offsetToUse[0],
        (Dimensions > 1 ? fillShape[Dimensions - 2] : 1) + offsetToUse[1],
        fillShape[Dimensions - 1] + offsetToUse[2]);

    auto fullAccessor = buffer.template get_access<sycl::access_mode::read>();
    T linear_id = 1;
    for (size_t z = offsetToUse[0]; z < shapeToCheck[0]; ++z) {
      for (size_t y = offsetToUse[1]; y < shapeToCheck[1]; ++y) {
        for (size_t x = offsetToUse[2]; x < shapeToCheck[2]; ++x) {
          auto value = accessHelper<Dimensions>(fullAccessor, z, y, x);
          ASSERT_EQ(linear_id, value);
          linear_id += 1;
        }
      }
    }

    sycl::id<3> adjustedFullShape(
        Dimensions > 2 ? fullShape[Dimensions - 3] : 1,
        Dimensions > 1 ? fullShape[Dimensions - 2] : 1,
        fullShape[Dimensions - 1]);

    for (size_t z = 0; z < adjustedFullShape[0]; ++z) {
      for (size_t y = 0; y < adjustedFullShape[1]; ++y) {
        for (size_t x = 0; x < adjustedFullShape[2]; ++x) {
          // Skip elements which we previously checked
          if (z >= offsetToUse[0] && z < shapeToCheck[0] &&
              y >= offsetToUse[1] && y < shapeToCheck[1] &&
              x >= offsetToUse[2] && x < shapeToCheck[2])
            continue;
          auto value = accessHelper<Dimensions>(fullAccessor, z, y, x);
          ASSERT_EQ(T{}, value) << "at (" << z << "; " << y << "; " << x << ")";
        }
      }
    }
  }

  template <int Dimensions, typename T = int>
  void checkFullCopyThroughIterator(const sycl::range<Dimensions> &shape) {
    std::vector<T> reference(shape.size());
    std::iota(reference.begin(), reference.end(), 0);
    sycl::buffer<T, Dimensions> buffer(reference.data(), shape);
    auto accessor = buffer.template get_access<sycl::access_mode::read_write>();

    ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIteratorImpl(
        reference, accessor.begin(), accessor.end()));
    ASSERT_NO_FATAL_FAILURE(checkFullCopyThroughIteratorImpl(
        reference, accessor.cbegin(), accessor.cend()));
  }

  template <int Dimensions, typename T = int>
  void
  checkPartialCopyThroughIterator(const sycl::range<Dimensions> &fullShape,
                                  const sycl::range<Dimensions> &copyShape,
                                  const sycl::id<Dimensions> &offset = {}) {
    std::vector<T> reference(fullShape.size());
    std::iota(reference.begin(), reference.end(), 0);
    sycl::buffer<T, Dimensions> buffer(reference.data(), fullShape);
    std::vector<T> copied;

    {
      auto accessor = buffer.template get_access<sycl::access_mode::read_write>(
          copyShape, offset);
      copied = copyThroughIterators<T>(accessor.begin(), accessor.end());
    }
    ASSERT_NO_FATAL_FAILURE(
        validatePartialCopyThroughIterator(copied, buffer, copyShape, offset));

    {
      auto accessor = buffer.template get_access<sycl::access_mode::read_write>(
          copyShape, offset);
      copied = copyThroughIterators<T>(accessor.cbegin(), accessor.cend());
    }
    ASSERT_NO_FATAL_FAILURE(
        validatePartialCopyThroughIterator(copied, buffer, copyShape, offset));
  }

private:
  template <typename IteratorT, typename T = int>
  void checkFullCopyThroughIteratorImpl(const std::vector<T> &reference,
                                        IteratorT begin, IteratorT end) {
    std::vector<T> copied = copyThroughIterators<T>(begin, end);

    ASSERT_EQ(copied.size(), reference.size());
    for (size_t i = 0, e = reference.size(); i < e; ++i) {
      ASSERT_EQ(copied[i], reference[i]);
    }
  }

  template <typename T, typename IteratorT>
  std::vector<T> copyThroughIterators(IteratorT begin, IteratorT end) {
    std::vector<T> copied;
    for (auto it = begin; it != end; ++it)
      copied.push_back(*it);

    return copied;
  }

  template <int Dimensions, typename T = int>
  void
  validatePartialCopyThroughIterator(const std::vector<T> &copied,
                                     sycl::buffer<T, Dimensions> &buffer,
                                     const sycl::range<Dimensions> &copyShape,
                                     const sycl::id<Dimensions> &offset = {}) {
    auto fullAccessor = buffer.template get_access<sycl::access_mode::read>();
    size_t linearId = 0;

    sycl::id<3> offsetToUse(Dimensions > 2 ? offset[Dimensions - 3] : 1,
                            Dimensions > 1 ? offset[Dimensions - 2] : 1,
                            offset[Dimensions - 1]);

    sycl::id<3> shapeToCheck(
        (Dimensions > 2 ? copyShape[Dimensions - 3] : 1) + offsetToUse[0],
        (Dimensions > 1 ? copyShape[Dimensions - 2] : 1) + offsetToUse[1],
        copyShape[Dimensions - 1] + offsetToUse[2]);

    for (size_t z = offsetToUse[0]; z < shapeToCheck[0]; ++z) {
      for (size_t y = offsetToUse[1]; y < shapeToCheck[1]; ++y) {
        for (size_t x = offsetToUse[2]; x < shapeToCheck[2]; ++x) {
          auto value = accessHelper<Dimensions>(fullAccessor, z, y, x);
          ASSERT_EQ(copied[linearId], value);
          ++linearId;
        }
      }
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
};

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
    // The test also checks the same for operator +, i.e. It + n
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

      ASSERT_EQ(It1, It2) << " with n = " << n;
      ASSERT_EQ(It + n, It2) << " with n = " << n;
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
    ASSERT_EQ(std::abs(It - It1), std::abs(It1 - It));
    ASSERT_EQ(std::abs(It - It2), std::abs(It2 - It));
    ASSERT_EQ(It1 - It, -3);
    ASSERT_EQ(It - It1, 3);
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

  { // It - n equivalent to:
    // iterator temp = It;
    // return temp -= n;
    auto It1 = accessor.end();
    auto It2 = accessor.end();
    const auto It3 = accessor.end();

    It2 -= 3;
    ASSERT_EQ(It1 - 3, It2);
    ASSERT_EQ(It1, accessor.end());
    // Check that operator-() can take a constant iterator
    ASSERT_EQ(It3 - 3, It2);
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
  ASSERT_NO_FATAL_FAILURE(
      checkPartialCopyThroughIterator(sycl::range<1>{10}, sycl::range<1>{5}));
  ASSERT_NO_FATAL_FAILURE(
      checkPartialCopyThroughIterator(sycl::range<1>{10}, sycl::range<1>{10}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithoutOffset2D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{5, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{2, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{5, 2}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 2}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithoutOffset3D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{5, 3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 5, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{5, 5, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{1, 2, 3}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithOffset1D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<1>{10}, sycl::range<1>{5}, sycl::id<1>{3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<1>{10}, sycl::range<1>{5}, sycl::id<1>{5}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithOffset2D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{10, 10}, sycl::range<2>{5, 5}, sycl::id<2>{3, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{10, 10}, sycl::range<2>{5, 5}, sycl::id<2>{3, 0}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{10, 10}, sycl::range<2>{5, 5}, sycl::id<2>{0, 3}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{10, 10}, sycl::range<2>{5, 5}, sycl::id<2>{5, 5}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{5, 10}, sycl::range<2>{3, 5}, sycl::id<2>{1, 4}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<2>{10, 5}, sycl::range<2>{5, 3}, sycl::id<2>{5, 1}));
}

TEST_F(AccessorIteratorTest, PartialCopyWithOffset3D) {
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{7, 7, 7}, sycl::range<3>{3, 3, 3}, sycl::id<3>{2, 2, 2}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{8, 8, 8}, sycl::range<3>{4, 4, 4}, sycl::id<3>{4, 4, 4}));
  // FIXME: figure out why the test below fails
  // ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
  //     sycl::range<3>{7, 7, 7}, sycl::range<3>{3, 3, 3}, sycl::id<3>{4, 4,
  //     4}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{7, 7, 7}, sycl::range<3>{3, 4, 5}, sycl::id<3>{3, 2, 1}));
  ASSERT_NO_FATAL_FAILURE(checkPartialCopyThroughIterator(
      sycl::range<3>{9, 8, 7}, sycl::range<3>{3, 4, 5}, sycl::id<3>{3, 2, 1}));
}

TEST_F(AccessorIteratorTest, FullWrite1D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<1>{10}, sycl::range<1>{10}, sycl::id<1>{0}));
}

TEST_F(AccessorIteratorTest, FullWrite2D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{5, 5}, sycl::id<2>{0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{2, 5}, sycl::range<2>{2, 5}, sycl::id<2>{0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 2}, sycl::range<2>{5, 2}, sycl::id<2>{0, 0}));
}

TEST_F(AccessorIteratorTest, FullWrite3D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{5, 5, 5}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{1, 5, 5}, sycl::range<3>{1, 5, 5}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 1, 5}, sycl::range<3>{5, 1, 5}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 1}, sycl::range<3>{5, 5, 1}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{3, 6, 4}, sycl::range<3>{3, 6, 4}, sycl::id<3>{0, 0, 0}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithoutOffset1D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<1>{10}, sycl::range<1>{5}, sycl::id<1>{0}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithoutOffset2D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 3}, sycl::id<2>{0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{2, 5}, sycl::range<2>{1, 3}, sycl::id<2>{0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 2}, sycl::range<2>{3, 1}, sycl::id<2>{0, 0}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithoutOffset3D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 3}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{1, 5, 5}, sycl::range<3>{0, 3, 3}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 1, 5}, sycl::range<3>{3, 1, 3}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 1}, sycl::range<3>{3, 3, 1}, sycl::id<3>{0, 0, 0}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{3, 6, 4}, sycl::range<3>{1, 3, 2}, sycl::id<3>{0, 0, 0}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithOffset1D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<1>{10}, sycl::range<1>{5}, sycl::id<1>{3}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithOffset2D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 5}, sycl::range<2>{3, 3}, sycl::id<2>{1, 1}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{3, 5}, sycl::range<2>{1, 3}, sycl::id<2>{1, 2}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<2>{5, 3}, sycl::range<2>{3, 1}, sycl::id<2>{1, 1}));
}

TEST_F(AccessorIteratorTest, PartialWriteWithOffset3D) {
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 5}, sycl::range<3>{3, 3, 3}, sycl::id<3>{1, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{3, 5, 5}, sycl::range<3>{0, 3, 3}, sycl::id<3>{1, 2, 2}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 2, 5}, sycl::range<3>{3, 1, 3}, sycl::id<3>{1, 1, 2}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{5, 5, 3}, sycl::range<3>{3, 3, 1}, sycl::id<3>{1, 1, 1}));
  ASSERT_NO_FATAL_FAILURE(checkWriteThroughIterator(
      sycl::range<3>{3, 6, 4}, sycl::range<3>{1, 3, 2}, sycl::id<3>{1, 3, 2}));
}
