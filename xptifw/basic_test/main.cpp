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
  test::utils::CommandLineParser options;

  options.addOption("--verbose")
      .setAbbreviation("-v")
      .setHelp("Run the tests in verbose mode. Running in this mode "
               "may\naffect performance test metrics.\n")
      .setRequired(false)
      .setType(test::utils::OptionType::String);

  options.addOption("--trace-points")
      .setAbbreviation("-t")
      .setHelp(
          "Number of trace points to use in the tests - Range [10-100000]\n")
      .setRequired(true)
      .setType(test::utils::OptionType::Integer);

  options.addOption("--type")
      .setAbbreviation("-y")
      .setHelp("Takes in the type of test to run. The options are:\n\n  o "
               "semantic\n  o performance\n\nSemantic tests will ignore all "
               "flags that are meant\nfor performance tests.\n")
      .setRequired(true)
      .setType(test::utils::OptionType::String);

  options.addOption("--test-id")
      .setAbbreviation("-i")
      .setHelp(
          "Takes in the test identifier to run a specific test. These\ntests "
          "will be identifiers within the semantic or performance tests.\n")
      .setRequired(true)
      .setType(test::utils::OptionType::Range);

  options.addOption("--num-threads")
      .setAbbreviation("-n")
      .setHelp("Number of threads to use to run the tests.\n")
      .setRequired(true)
      .setType(test::utils::OptionType::Range);

  options.addOption("--overhead")
      .setAbbreviation("-o")
      .setHelp("Overhead limit in percentage - Range[0.1-15]\n")
      .setRequired(false)
      .setType(test::utils::OptionType::Float);

  options.addOption("--report")
      .setAbbreviation("-r")
      .setHelp("Print the results in tabular form.\n")
      .setRequired(false)
      .setType(test::utils::OptionType::String);

  options.addOption("--tp-frequency")
      .setAbbreviation("-f")
      .setHelp("Trace point creation frequency as a percentage of tracepoint "
               "instances-Range [1-100]\n")
      .setRequired(false)
      .setType(test::utils::OptionType::Float);

  options.parse(argc, argv);

  test::semantic::TestCorrectness ct(options);
  test::performance::TestPerformance pt(options);

  ct.run();
  pt.run();
}
