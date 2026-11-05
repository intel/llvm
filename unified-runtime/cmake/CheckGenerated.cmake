# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Regenerates the source from the specification and fails if that changed any
# tracked file or created any file that is not tracked by git.
#
# Usage:
#   cmake -DSOURCE_DIR=<UR source dir> -DBINARY_DIR=<build dir>
#         -P CheckGenerated.cmake

function(get_untracked_files out)
    execute_process(
        COMMAND git ls-files --others --exclude-standard
        WORKING_DIRECTORY ${SOURCE_DIR}
        OUTPUT_VARIABLE files
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" files "${files}")
    set(${out} "${files}" PARENT_SCOPE)
endfunction()

# Untracked files that existed before generation are not reported, so local
# scratch files don't make the check fail.
get_untracked_files(untracked_before)

execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target generate-code
    COMMAND_ERROR_IS_FATAL ANY)

execute_process(
    COMMAND git diff --exit-code
    WORKING_DIRECTORY ${SOURCE_DIR}
    RESULT_VARIABLE diff_result)

get_untracked_files(untracked_new)
if(untracked_before)
    list(REMOVE_ITEM untracked_new ${untracked_before})
endif()

if(untracked_new)
    list(JOIN untracked_new "\n  " untracked_new)
    message(SEND_ERROR "The generator created files that are not tracked by "
        "git. Add them to the repository or fix the output path in the "
        "generator:\n  ${untracked_new}")
endif()
if(NOT diff_result EQUAL 0)
    message(SEND_ERROR "The generated source differs from the committed "
        "source. Run the 'generate' target and commit the result.")
endif()
