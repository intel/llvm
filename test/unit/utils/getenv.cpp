// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include <map>

#include <gtest/gtest.h>

#include "helpers.h"
#include "ur_util.hpp"

class GetEnvToMapWithParam
    : public ::testing::TestWithParam<
          std::map<std::string, std::vector<std::string>>> {
protected:
  int ret = -1;
  std::map<std::string, std::vector<std::string>> test_map;

  void SetUp() override {
    std::stringstream env_var_value;
    test_map = GetParam();

    for (auto const &item : test_map) {
      if (&item.first != &test_map.begin()->first) {
        env_var_value << ";";
      }
      env_var_value << item.first << ":";
      for (auto const &vec_value : item.second) {
        env_var_value << vec_value;
        if (&vec_value != &item.second.back()) {
          env_var_value << ",";
        }
      }
    }

    ret = setenv("UR_TEST_ENV_VAR", env_var_value.str().c_str(), 1);
    ASSERT_EQ(ret, 0);
  }

  void TearDown() override {
    ret = unsetenv("UR_TEST_ENV_VAR");
    ASSERT_EQ(ret, 0);
  }
};

using TestMaps = std::vector<std::map<std::string, std::vector<std::string>>>;

TestMaps getTestMaps() {
  TestMaps test_maps_vec;
  test_maps_vec.push_back({{"param_1", {"value_1", "value_2"}},
                           {"param_2", {"value_1", "value_3"}}});
  test_maps_vec.push_back({{"param_1", {"value_1"}}, {"param_2", {"value_1"}}});
  test_maps_vec.push_back({{"param_1", {"value_1"}}});
  return test_maps_vec;
}

INSTANTIATE_TEST_SUITE_P(possibleEnvVarValues, GetEnvToMapWithParam,
                         testing::ValuesIn(getTestMaps()));

TEST_P(GetEnvToMapWithParam, MapValues) {
  auto map = getenv_to_map("UR_TEST_ENV_VAR");
  ASSERT_TRUE(map.has_value());
  for (auto &it : test_map) {
    ASSERT_EQ(map->at(it.first), test_map[it.first]);
  }
}

TEST(GetenvToVec, OneValue) {
  int ret = setenv("UR_TEST_ENV_VAR", "value_1", 1);
  ASSERT_EQ(ret, 0);

  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_TRUE(vec.has_value());
  ASSERT_EQ(vec->front(), "value_1");

  ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);
}

TEST(GetenvToVec, OneValueSpecialChars) {
  int ret = setenv("UR_TEST_ENV_VAR", "!@#$%^&*()_+-={}[]|\\\\\"<>?/`~", 1);
  ASSERT_EQ(ret, 0);

  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_TRUE(vec.has_value());
  ASSERT_EQ(vec->front(), "!@#$%^&*()_+-={}[]|\\\\\"<>?/`~");

  ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);
}

TEST(GetenvToVec, HugeInput) {
  std::string huge_str;
  huge_str.reserve(1024);
  for (int i = 0; i < 102; ++i) {
    huge_str += "huuuuuuuge";
  }

  int ret = setenv("UR_TEST_ENV_VAR", huge_str.c_str(), 1);
  ASSERT_EQ(ret, 0);

  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_TRUE(vec.has_value());
  ASSERT_EQ(vec->front(), huge_str);

  ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);
}

TEST(GetenvToVec, MultipleValues) {
  int ret = setenv("UR_TEST_ENV_VAR", "value_1,value_2", 1);
  ASSERT_EQ(ret, 0);

  std::vector<std::string> test_vec = {"value_1", "value_2"};
  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_TRUE(vec.has_value());
  ASSERT_EQ(vec, test_vec);

  ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);
}

TEST(GetenvToVec, EmptyEnvVar) {
  int ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);

  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_FALSE(vec.has_value());
}

TEST(GetenvToMap, EmptyEnvVar) {
  int ret = unsetenv("UR_TEST_ENV_VAR");
  ASSERT_EQ(ret, 0);

  auto map = getenv_to_map("UR_TEST_ENV_VAR");
  ASSERT_FALSE(map.has_value());
}

// ////////////////////////////////////////////////////////////////////////////////////
// // Negative tests

class GetEnvFailureWithParam : public ::testing::TestWithParam<std::string> {
protected:
  int ret = -1;
  std::string env_var_value = "";

  void SetUp() override {
    env_var_value = GetParam();
    ret = setenv("UR_TEST_ENV_VAR", env_var_value.c_str(), 1);
    ASSERT_EQ(ret, 0);
  }

  void TearDown() override {
    ret = unsetenv("UR_TEST_ENV_VAR");
    ASSERT_EQ(ret, 0);
  }
};

class GetenvToMap : public GetEnvFailureWithParam {};

class GetenvToVec : public GetEnvFailureWithParam {};

INSTANTIATE_TEST_SUITE_P(
    WrongValuesForMap, GetenvToMap,
    testing::Values("value_1;value_2", "param_1:value_1,value_2;param_2:",
                    "param_1:value_1,value_2;:value_1", ",;:", ",;",
                    "rvrawerv)(*&)($@#93939854;;)", "simple", ",", ":", ";",
                    "value,value", "param:value;param:value",
                    "param,value;param:value", "param:value;param_2,value",
                    "param:value;param_2", "param:value;param_2;param_3",
                    "param:value:value_2", "param:value;:", "param:value;,"));

TEST_P(GetenvToMap, WrongEnvVarValues) {
  ASSERT_THROW(getenv_to_map("UR_TEST_ENV_VAR"), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(WrongValuesForVec, GetenvToVec,
                         testing::Values("value_1;value_2",
                                         "param_1:value_1,value_2;param_2", ",",
                                         ";", ":", ",,,",
                                         "rvrawerv)(*&)($@#93939854,,)",
                                         "value,,", "value;"));

TEST_P(GetenvToVec, WrongEnvVarValues) {
  ASSERT_THROW(getenv_to_vec("UR_TEST_ENV_VAR"), std::invalid_argument);
}

#if defined(_WIN32)
TEST(GetEnvExceptionWindows, HugeInput) {
  std::string huge_str;
  huge_str.reserve(1040);
  for (int i = 0; i < 103; ++i) {
    huge_str += "huuuuuuuge";
  }

  auto map = getenv_to_map("UR_TEST_ENV_VAR");
  ASSERT_FALSE(map.has_value());
  auto vec = getenv_to_vec("UR_TEST_ENV_VAR");
  ASSERT_FALSE(vec.has_value());
}
#endif
