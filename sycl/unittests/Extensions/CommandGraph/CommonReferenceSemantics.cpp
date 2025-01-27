//==----------------------- CommandGraph.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

#include <unordered_map>

using namespace sycl;
using namespace sycl::ext::oneapi;

/**
 * Checks that the operators and constructors of graph related classes meet the
 * common reference semantics.
 * @param objFactory A function object that returns an object to be tested.
 */
template <typename T, typename LambdaType>
void testSemantics(LambdaType &&objFactory) {

  T Obj1 = objFactory();
  T Obj2 = objFactory();

  // Check the == and != operators.
  ASSERT_FALSE(Obj1 == Obj2);
  ASSERT_TRUE(Obj1 != Obj2);

  // Check Copy Constructor and Assignment operators.
  T Obj1Copy = Obj1;
  T Obj2CopyConstructed(Obj2);
  ASSERT_TRUE(Obj1Copy == Obj1);
  ASSERT_TRUE(Obj2CopyConstructed == Obj2);

  // Check Move Constructor and Move Assignment operators.
  auto Obj1Move = std::move(Obj1);
  auto Obj2MoveConstructed(std::move(Obj2));
  ASSERT_TRUE(Obj1Move == Obj1Copy);
  ASSERT_TRUE(Obj2MoveConstructed == Obj2CopyConstructed);
}

TEST_F(CommandGraphTest, ModifiableGraphSemantics) {
  sycl::queue Queue;
  auto factory = [&]() {
    return experimental::command_graph(Queue.get_context(), Queue.get_device());
  };

  ASSERT_NO_FATAL_FAILURE(
      testSemantics<
          experimental::command_graph<experimental::graph_state::modifiable>>(
          factory));
}

TEST_F(CommandGraphTest, ExecutableGraphSemantics) {
  sycl::queue Queue;

  auto factory = [&]() {
    experimental::command_graph Graph(Queue.get_context(), Queue.get_device());
    return Graph.finalize();
  };
  ASSERT_NO_FATAL_FAILURE(
      testSemantics<
          experimental::command_graph<experimental::graph_state::executable>>(
          factory));
}

TEST_F(CommandGraphTest, NodeSemantics) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto factory = [&]() {
    return Graph.add(
        [&](handler &CGH) { CGH.parallel_for(1, [=](item<1> Item) {}); });
  };
  ASSERT_NO_FATAL_FAILURE(testSemantics<experimental::node>(factory));
}

TEST_F(CommandGraphTest, DynamicCGSemantics) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto CGF = [&](handler &CGH) { CGH.parallel_for(1, [=](item<1> Item) {}); };

  auto factory = [&]() {
    return experimental::dynamic_command_group(Graph, {CGF});
  };
  ASSERT_NO_FATAL_FAILURE(
      testSemantics<experimental::dynamic_command_group>(factory));
}

TEST_F(CommandGraphTest, DynamicParamSemantics) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto factory = [&]() {
    return experimental::dynamic_parameter<int>(Graph, 1);
  };
  ASSERT_NO_FATAL_FAILURE(
      testSemantics<experimental::dynamic_parameter<int>>(factory));
}

/**
 * Checks for potential hash collisions in the hash implementations of graph
 * related classes.
 * @param objFactory A function object that returns an object to be tested.
 */
template <typename T, typename LambdaType>
void testHash(LambdaType &&objFactory) {

  const int NumObjects = 100;

  std::unordered_set<size_t> HashSet{};

  T Obj1 = objFactory();
  T Obj2 = objFactory();
  T Obj3 = objFactory();
  T Obj4 = objFactory();

  ASSERT_TRUE(HashSet.insert(std::hash<T>{}(Obj1)).second);
  ASSERT_TRUE(HashSet.insert(std::hash<T>{}(Obj2)).second);

  // Create objects and destroy them immediately to confirm that the
  // hashes are unique and are not reused.
  for (int i = 0; i < NumObjects; ++i) {
    T ObjI = objFactory();
    ASSERT_TRUE(HashSet.insert(std::hash<T>{}(ObjI)).second);
  }

  ASSERT_TRUE(HashSet.insert(std::hash<T>{}(Obj3)).second);
  ASSERT_TRUE(HashSet.insert(std::hash<T>{}(Obj4)).second);

  ASSERT_TRUE(HashSet.size() == (NumObjects + 4));
}

TEST_F(CommandGraphTest, ModifiableGraphHash) {
  sycl::queue Queue;
  auto factory = [&]() {
    return experimental::command_graph(Queue.get_context(), Queue.get_device());
  };

  ASSERT_NO_FATAL_FAILURE(
      testHash<
          experimental::command_graph<experimental::graph_state::modifiable>>(
          factory));
}

TEST_F(CommandGraphTest, ExecutableGraphHash) {
  sycl::queue Queue;

  auto factory = [&]() {
    experimental::command_graph Graph(Queue.get_context(), Queue.get_device());
    return Graph.finalize();
  };
  ASSERT_NO_FATAL_FAILURE(
      testHash<
          experimental::command_graph<experimental::graph_state::executable>>(
          factory));
}

TEST_F(CommandGraphTest, NodeHash) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto factory = [&]() {
    return Graph.add(
        [&](handler &CGH) { CGH.parallel_for(1, [=](item<1> Item) {}); });
  };
  ASSERT_NO_FATAL_FAILURE(testHash<experimental::node>(factory));
}

TEST_F(CommandGraphTest, DynamicCommandGroupHash) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto CGF = [&](handler &CGH) { CGH.parallel_for(1, [=](item<1> Item) {}); };

  auto factory = [&]() {
    return experimental::dynamic_command_group(Graph, {CGF});
  };
  ASSERT_NO_FATAL_FAILURE(
      testHash<experimental::dynamic_command_group>(factory));
}

TEST_F(CommandGraphTest, DynamicParameterHash) {
  sycl::queue Queue;
  experimental::command_graph Graph(Queue.get_context(), Queue.get_device());

  auto factory = [&]() {
    return experimental::dynamic_parameter<int>(Graph, 1);
  };
  ASSERT_NO_FATAL_FAILURE(
      testHash<experimental::dynamic_parameter<int>>(factory));
}
