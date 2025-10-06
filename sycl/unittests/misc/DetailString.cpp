#include <gtest/gtest.h>
#include <sycl/detail/string.hpp>

class SYCLDetailStringTest : public ::testing::Test {};

TEST_F(SYCLDetailStringTest, DefaultConstructor) {
  sycl::detail::string empty_s;
  EXPECT_TRUE(empty_s.empty());
  EXPECT_STREQ(empty_s.c_str(), "");
}

TEST_F(SYCLDetailStringTest, StringViewConstructor) {
  std::string_view sv = "Hello, World!";
  sycl::detail::string s1(sv);
  EXPECT_STREQ(s1.c_str(), "Hello, World!");
}

TEST_F(SYCLDetailStringTest, CopyConstructor) {
  sycl::detail::string s1("Hello, World!");
  sycl::detail::string s2(s1);
  EXPECT_STREQ(s2.c_str(), "Hello, World!");

  // Check for deep copy: modifying the original should not affect the copy.
  s1 = "Changed";
  EXPECT_STREQ(s2.c_str(), "Hello, World!");
}

TEST_F(SYCLDetailStringTest, MoveConstructor) {
  sycl::detail::string s1("Changed");
  sycl::detail::string s2(std::move(s1));
  EXPECT_STREQ(s2.c_str(), "Changed");
}

TEST_F(SYCLDetailStringTest, StringViewAssignment) {
  sycl::detail::string s;
  s = "New String";
  EXPECT_STREQ(s.c_str(), "New String");
  std::string_view sv = "From String View";
  s = sv;
  EXPECT_STREQ(s.c_str(), "From String View");
}

TEST_F(SYCLDetailStringTest, CopyAssignment) {
  sycl::detail::string s1("Hello, World!");
  sycl::detail::string s2;
  s2 = s1;
  EXPECT_STREQ(s2.c_str(), "Hello, World!");

  // Check for deep copy.
  s1 = "Changed";
  EXPECT_STREQ(s2.c_str(), "Hello, World!");
}

TEST_F(SYCLDetailStringTest, MoveAssignment) {
  sycl::detail::string s1("Changed");
  sycl::detail::string s2;
  s2 = std::move(s1);
  EXPECT_STREQ(s2.c_str(), "Changed");
}

TEST_F(SYCLDetailStringTest, Methods) {
  sycl::detail::string s_not_empty("not empty");
  sycl::detail::string s_empty;
  // Test c_str() and data().
  EXPECT_STREQ(s_not_empty.data(), "not empty");
  EXPECT_STREQ(s_not_empty.c_str(), "not empty");
  EXPECT_EQ(s_not_empty.data(), s_not_empty.c_str());

  // Test empty(). Like really.
  EXPECT_FALSE(s_not_empty.empty());
  EXPECT_TRUE(sycl::detail::string("").empty());
  EXPECT_TRUE(sycl::detail::string().empty());
  EXPECT_TRUE(s_empty.empty());
}

TEST_F(SYCLDetailStringTest, Swap) {
  sycl::detail::string s1("first");
  sycl::detail::string s2("second");
  swap(s1, s2);
  EXPECT_STREQ(s1.c_str(), "second");
  EXPECT_STREQ(s2.c_str(), "first");
}

TEST_F(SYCLDetailStringTest, ComparisonOperators) {
  sycl::detail::string s("match");
  std::string_view sv_match("match");
  std::string_view sv_no_match("no match");

  EXPECT_EQ(s, sv_match);
  EXPECT_EQ(sv_match, s);
  EXPECT_FALSE(s == sv_no_match);
}
