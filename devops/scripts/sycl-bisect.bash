#!/bin/bash

# This is a script for running git-bisect on sycl, sycl-web, and sycl pulldown
# branches. This is mostly a standard git-bisect, except that any upstream
# commits that aren't sycl-based must be merged to a sycl-based branch before
# being tested. When possible, this is done in a secondary worktree to avoid
# clobbering files, which enables incremental builds and significantly improves
# build times nearer to the end of the bisection.

# Parse options.
OPTS=$(getopt -n sycl-bisect-test-commit.bash -o 'ht:c:b' -l 'help,test:,command:,command-allow-bisect-codes' -- "$@")
if [[ $? != 0 ]]; then
  exit 1
fi
eval set -- "$OPTS"
unset OPTS TEST COMMAND COMMAND_ALLOW_BISECT_CODES
while true; do
  case "$1" in
    '-h'|'--help')
      cat <<HELP
Usage:

$0 [opts] <bad commit> <good commits...>

  -h,--help                       Print this help message
  -t,--test <test>                Test commits with LIT test <test>, a test file
                                  path relative to the current working directory
  -c,--command <command>          Test commits with command <command>, which
                                  will be executed for each commit via bash -c
                                  in the current working directory
  -b,--command-allow-bisect-codes Pass 125 and >127 return codes to git-bisect
                                  directly when using --command to allow the
                                  command to skip commits or stop bisection. If
                                  this option is not present, any non-zero
                                  return value is interpreted as a bad commit.

Examples:

"Clang :: SemaSYCL/accessor_inheritance.cpp" started failing on sycl-web because
of one of the commits merged by 96f730774ac4. To find which one of these 96
commits caused this test failure, sycl-bisect.bash can be run like this:

$ devops/scripts/sycl-bisect.bash 96f730774ac4 96f730774ac4^ --test clang/test/SemaSYCL/accessor_inheritance.cpp
HELP
      exit 0
      ;;
    '-t'|'--test')
      export TEST="$2"
      shift 2
      ;;
    '-c'|'--command')
      export COMMAND="$2"
      shift 2
      ;;
    '-b'|'--command-allow-bisect-codes')
      export COMMAND_ALLOW_BISECT_CODES=1
      shift
      ;;
    '--')
      shift
      break
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      exit 1
      ;;
  esac
done

# Validate options.
if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
  echo "A bad rev and at least one good rev must be passed" >&2
  exit 1
fi
if [[ "$TEST" == "" ]] && [[ "$COMMAND" == "" ]]; then
  echo "--test <test> or --command <command> must be used to specify a test" >&2
  exit 1
fi
if [[ "$TEST" != "" ]] && [[ "$COMMAND" != "" ]]; then
  echo "Only one of --test <test> or --command <command> is allowed" >&2
  exit 1
fi
if [[ "$COMMAND_ALLOW_BISECT_CODES" != "" ]] && [[ "$COMMAND" == "" ]]; then
  echo "--command-allow-bisect-codes is only allowed with --command <command>" >&2
  exit 1
fi

# Save the current working directory before switching to the repository's
# top-level directory. Tests will be run relative to the original working
# directory.
export TEST_WD="$PWD"
TOPLEVEL="$(git rev-parse --show-toplevel)"
if [[ "$TOPLEVEL" == "" ]]; then
  exit 1
fi
cd "$TOPLEVEL"

# A commit that should be on all sycl-based branches this script works with.
# This is set to the initial sycl-specific commit.
export SYCL_ANCESTOR=1e0b4966ba9a

# This commit should be in the repository.
if ! git rev-parse $SYCL_ANCESTOR &>/dev/null; then
  echo "Current repository is not sycl-based" >&2
  exit 1
fi

# Along with all the bad/good revs.
for REV in "$@"; do
  if ! git rev-parse --verify "$REV" &>/dev/null; then
    echo "'$REV' is not a valid revision" >&2
    exit 1
  fi
done

# Make sure the build directory is set up; if not, tell the user to run
# the configure step.
if [[ ! -f build/CMakeCache.txt ]]; then
  echo "The build directory doesn't seem to be configured yet." >&2
  echo "Please run the configure step as documented in sycl/doc/GetStartedGuide.md:" >&2
  echo "  https://intel.github.io/llvm/GetStartedGuide.html#build-dpc-toolchain" >&2
  exit 1
fi

# If this is a LIT test, make sure FileCheck and other testing utilities are
# built.
if [[ ! -f build/bin/FileCheck ]]; then
  echo "FileCheck not found; building test-depends"
  cmake --build build -- test-depends -j $(nproc)
fi

# Determines if the passed commit is sycl-based or not.
function is_sycl_based_commit {
  git merge-base --is-ancestor $SYCL_ANCESTOR "$1" 2>/dev/null
}

# Checks if this commit/branch is a candidate for SYCL_DESCENDANT. For that to
# be the case, it needs to be sycl-based and a descendant of the bad and good
# commits.
readonly -a REQUIRED_ANCESTORS=($SYCL_ANCESTOR "$@")
function check_sycl_descendant {
  local ancestor
  for ancestor in "${REQUIRED_ANCESTORS[@]}"; do
    git merge-base --is-ancestor "$ancestor" "$1" &>/dev/null || return 1
  done
  export SYCL_DESCENDANT="$1"
  return 0
}

# If SYCL_DESCENDANT isn't specified, poke around until a suitable commit/branch
# is found.
function find_sycl_descendant {
  [[ "$SYCL_DESCENDANT" != "" ]] && return

  # Try the bad rev first.
  check_sycl_descendant "${REQUIRED_ANCESTORS[1]}" && return

  # Try "sycl" and "sycl-web", both locally and in all the remotes.
  check_sycl_descendant "sycl" && return
  check_sycl_descendant "sycl-web" && return
  local remote
  for remote in $(git remote); do
    check_sycl_descendant "$remote/sycl" && return
    check_sycl_descendant "$remote/sycl-web" && return
  done

  # Try all of the local branches.
  local branch
  for branch in $(git branch); do
    check_sycl_descendant "$branch" && return
  done

  # Give up and ask the user to pick one.
  echo "Unable to find a sycl-based branch containing all bad/good commits" >&2
  echo "Please specify one with SYCL_DESCENDANT" >&2
  exit 1
}
find_sycl_descendant

# Try creating a worktree for out-of-tree merges. If this is successful, set a
# trap to clean it up when bisection is complete.
echo "Attempting to set up sycl-bisect-merge worktree..."
git worktree add sycl-bisect-merge \
  && trap "git worktree remove sycl-bisect-merge" EXIT

# Save test-commit-sycl-bisect.bash to a temporary file to use during the
# bisection. Otherwise, git-bisect might check out a commit with a substantially
# different version than sycl-bisect.bash expects, or it might check out a
# commit where test-commit-sycl-bisect.bash doesn't exist at all.
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
TEST_COMMIT="$(mktemp --tmpdir test-commit-XXX.bash)"
if [[ -f "$TEST_COMMIT" ]]; then
  if [[ "$(trap -p EXIT)" != "" ]]; then
    trap "git worktree remove sycl-bisect-merge ; rm $TEST_COMMIT" EXIT
  else
    trap "rm $TEST_COMMIT" EXIT
  fi
  cp "$SCRIPT_DIR/test-commit-sycl-bisect.bash" "$TEST_COMMIT"
else
  TEST_COMMIT="$SCRIPT_DIR/test-commit-sycl-bisect.bash"
fi

# Do the bisection.
git bisect start --no-checkout "$@" || exit
git bisect run bash "$TEST_COMMIT"
git bisect reset
