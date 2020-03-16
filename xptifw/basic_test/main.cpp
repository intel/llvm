//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
#include "cl_processor.hpp"

// This test will expose the correctness and performance tests
// through the command-line options.
int main(int argc, char **argv) {
  test::utils::cl_parser options;

  options.add_option("--verbose")
      .set_abbreviation("-v")
      .set_help("Run the tests in verbose mode. Running in this mode "
                "may\naffect performance test metrics.\n")
      .set_required(false)
      .set_type(test::utils::OptionType::String);

  options.add_option("--trace-points")
      .set_abbreviation("-t")
      .set_help(
          "Number of trace points to use in the tests - Range [10-100000]\n")
      .set_required(true)
      .set_type(test::utils::OptionType::Integer);

  options.add_option("--type")
      .set_abbreviation("-y")
      .set_help("Takes in the type of test to run. The options are:\n\n  o "
                "semantic\n  o performance\n\nSemantic tests will ignore all "
                "flags that are meant\nfor performance tests.\n")
      .set_required(true)
      .set_type(test::utils::OptionType::String);

  options.add_option("--test-id")
      .set_abbreviation("-i")
      .set_help(
          "Takes in the test identifier to run a specific test. These\ntests "
          "will be identifiers within the semantic or performance tests.\n")
      .set_required(true)
      .set_type(test::utils::OptionType::Range);

  options.add_option("--num-threads")
      .set_abbreviation("-n")
      .set_help("Number of threads to use to run the tests.\n")
      .set_required(true)
      .set_type(test::utils::OptionType::Range);

  options.add_option("--overhead")
      .set_abbreviation("-o")
      .set_help("Overhead limit in percentage - Range[0.1-15]\n")
      .set_required(false)
      .set_type(test::utils::OptionType::Float);

  options.add_option("--report")
      .set_abbreviation("-r")
      .set_help("Print the results in tabular form.\n")
      .set_required(false)
      .set_type(test::utils::OptionType::String);

  options.add_option("--tp-frequency")
      .set_abbreviation("-f")
      .set_help("Trace point creation frequency as a percentage of tracepoint "
                "instances-Range [1-100]\n")
      .set_required(false)
      .set_type(test::utils::OptionType::Float);

  options.parse(argc, argv);

  test::semantic::test_correctness ct(options);
  test::performance::test_performance pt(options);

  ct.run();
  pt.run();
}
