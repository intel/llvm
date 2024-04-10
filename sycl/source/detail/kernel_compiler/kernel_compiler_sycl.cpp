//==-- kernel_compiler_opencl.cpp  OpenCL kernel compilation support       -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/exception.hpp> // make_error_code

#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>

#include "kernel_compiler_sycl.hpp"

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

std::string generate_semi_unique_id() {
  // Get the current time as a time_t object
  std::time_t now = std::time(nullptr);

  // Convert time_t to a string with format YYYYMMDD_HHMMSS
  std::tm *local_time = std::localtime(&now);
  std::stringstream ss;
  ss << std::put_time(local_time, "%Y%m%d_%H%M%S");

  // amend with random number
  std::random_device rd;
  int random_number = rd() % 900 + 100;
  ss << "_" << std::setfill('0') << std::setw(3)
     << random_number; // Pad with leading zeros

  return ss.str();
}

std::filesystem::path prepare_ws(const std::string &id) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();

  std::filesystem::path new_directory_path = tmp / id;

  try {
    std::filesystem::create_directories(new_directory_path);
  } catch (std::filesystem::filesystem_error const &e) {
    throw sycl::exception(sycl::errc::build, e.what());
  }

  // CP
  std::cout << "Directory created: " << new_directory_path << std::endl;

  return new_directory_path;
}

void output_preamble(std::ofstream &os, const std::filesystem::path &file_path,
                     const std::string &id) {

  os << "/*\n  clang++ -fsycl -o " << id << ".bin -fsycl-dump-device-code=./ "
     << id << ".cpp \n */" << std::endl;
}

std::filesystem::path output_cpp(const std::filesystem::path &parent_dir,
                                 const std::string &id,
                                 std::string raw_code_string) {
  std::filesystem::path file_path = parent_dir / (id + ".cpp");
  std::ofstream outfile(file_path, std::ios::out | std::ios::trunc);

  if (outfile.is_open()) {
    output_preamble(outfile, file_path, id);
    outfile << raw_code_string << std::endl;

    // temporarily needed until -c works with -fsycl-dump-spirv
    outfile << "int main(){ return 0; }" << std::endl;

    outfile.close(); // Close the file when finished
  } else {
    throw sycl::exception(sycl::errc::build,
                          "Failed to open .cpp file for write: " +
                              file_path.string());
  }
  return file_path;
}

void output_include_files(const std::filesystem::path &dpath,
                          include_pairs_t IncludePairs) {
  using pairStrings = std::pair<std::string, std::string>;
  for (pairStrings p : IncludePairs) {
    std::filesystem::path file_path = dpath / p.first;
    std::ofstream outfile(file_path, std::ios::out | std::ios::trunc);
    if (outfile.is_open()) {
      outfile << p.second << std::endl;

      outfile.close();
    } else {
      throw sycl::exception(sycl::errc::build,
                            "Failed to open include file for write: " +
                                file_path.string());
    }
  }
}

void invoke_compiler(const std::filesystem::path &fpath,
                     const std::filesystem::path &dpath, const std::string &id,
                     const std::vector<std::string> &UserArgs,
                     std::string *LogPtr) {

  std::filesystem::path file_path(fpath);
  std::filesystem::path parent_dir(dpath);
  std::filesystem::path target_path = parent_dir / (id + ".bin");
  std::filesystem::path log_path = parent_dir / "compilation_log.txt";
#ifdef __WIN32
  std::string compiler = "clang++.exe";
#else
  std::string compiler = "clang++";
#endif

  // TODO: UserArgs!!!

  std::string command =
      compiler + " -fsycl -o " + target_path.make_preferred().string() +
      " -fsycl-dump-device-code=" + parent_dir.make_preferred().string() + " " +
      file_path.make_preferred().string() + " 2> " +
      log_path.make_preferred().string();

  // CP
  std::cout << "command: " << command << std::endl;

  int result = std::system(command.c_str());

  // Read the log file contents into the log variable
  std::string CompileLog;
  std::ifstream log_stream;
  log_stream.open(log_path);
  if (log_stream.is_open()) {
    std::stringstream log_buffer;
    log_buffer << log_stream.rdbuf();
    CompileLog.append(log_buffer.str());
    if (LogPtr != nullptr)
      LogPtr->append(log_buffer.str());

    // CP
    std::cout << "compile log: " << CompileLog << std::endl;
  } else if (result == 0 && LogPtr != nullptr) {
    // if there was a compilation problem, we want to report that (below)
    // not a mere "missing log" error.
    throw sycl::exception(sycl::errc::build,
                          "failure retrieving compilation log");
  }

  if (result != 0) {
    throw sycl::exception(sycl::errc::build,
                          "Compile failure: " + std::to_string(result) + " " +
                              CompileLog);
  }
}

std::filesystem::path find_spv(const std::filesystem::path &parent_dir,
                               const std::string &id) {
  std::regex pattern_regex(id + R"(.*\.spv)");

  // Iterate through all files in the directory matching the pattern
  for (const auto &entry : std::filesystem::directory_iterator(parent_dir)) {
    if (entry.is_regular_file() &&
        std::regex_match(entry.path().filename().string(), pattern_regex)) {
      // CP
      std::cout << "Matching file found: " << entry.path() << std::endl;
      return entry.path();
    }
  }
  // File not found, throw
  throw sycl::exception(sycl::errc::build, "SPIRV output matching " + id +
                                               " missing from " +
                                               parent_dir.filename().string());
}

spirv_vec_t load_spv_from_file(std::filesystem::path file_name) {
  std::ifstream spv_stream(file_name, std::ios::binary);
  spv_stream.seekg(0, std::ios::end);
  size_t sz = spv_stream.tellg();
  spv_stream.seekg(0);
  spirv_vec_t spv(sz);
  spv_stream.read(reinterpret_cast<char *>(spv.data()), sz);

  return spv;
}

spirv_vec_t SYCL_to_SPIRV(const std::string &SYCLSource,
                          include_pairs_t IncludePairs,
                          const std::vector<std::string> &UserArgs,
                          std::string *LogPtr) {
  // clang-format off
  const std::string id                    = generate_semi_unique_id();
  const std::filesystem::path parent_dir  = prepare_ws(id);
  std::filesystem::path file_path         = output_cpp(parent_dir, id, SYCLSource);
                                            output_include_files(parent_dir, IncludePairs);
                                            invoke_compiler(file_path, parent_dir, id, UserArgs, LogPtr);
  std::filesystem::path spv_path          = find_spv(parent_dir, id);
                                     return load_spv_from_file(spv_path);
  // clang-format on
}

bool SYCL_Compilation_Available() {
  // check for clang++ clang++.exe icpx icpx.exe on PATH

  return true;
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl