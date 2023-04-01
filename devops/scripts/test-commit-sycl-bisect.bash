#!/bin/bash

# This is a helper script for sycl-bisect.bash to test individual commits. It
# must be run in the repository's top-level directory, and relies on some
# environment variables set by sycl-bisect.bash to carry out merges and testing:
#
# SYCL_ANCESTOR: A ref on the sycl branch that is an ancestor of all of the
#  sycl-based commits in the bisect range; this is required for merging upstream
#  commits.
# SYCL_DESCENDANT: A ref descended from both $SYCL_ANCESTOR and BISECT_HEAD;
#   this is required for finding the right point to merge upstream commits.
# TEST_WD: The original working directory of sycl-bisect.bash, which should be
#   used as the base directory for TEST or working directory for COMMAND.
# TEST: A LIT test to run, if COMMAND is not set.
# COMMAND: A bash command to run, if TEST is not set.
# COMMAND_ALLOW_BISECT_CODES: If this is set, return codes from COMMAND are
#   passed through directly to sycl-bisect so they can be used to skip commits
#   or stop bisection.

# Indicates to git-bisect to skip this commit
function bad_commit { exit 125; }

# Indicates to git-bisect that it should exit
function stop_bisect { exit 128; }

# Sanity check the environment variables and working directory.
if [[ "$TEST" == "" ]] && [[ "$COMMAND" == "" ]]; then
  echo "TEST or COMMAND must be set" >&2
  stop_bisect
fi
if [[ "$TEST" != "" ]] && [[ "$COMMAND" != "" ]]; then
  echo "TEST and COMMAND must not both be set" >&2
  stop_bisect
fi
if [[ "$TEST_WD" == "" ]]; then
  echo "TEST_WD must be set" >&2
  stop_bisect
fi
if [[ "$SYCL_ANCESTOR" == "" ]]; then
  echo "SYCL_ANCESTOR must be set" >&2
  stop_bisect
fi
TOPLEVEL="$(git rev-parse --show-toplevel)"
if [[ "$(readlink -f .)" != "$TOPLEVEL" ]]; then
  echo "Not at top-level directory of repository" >&2
  stop_bisect
fi
if ! git rev-parse BISECT_HEAD &>/dev/null; then
  echo "No BISECT_HEAD found; make sure you're using git-bisect's --no-checkout mode" >&2
  stop_bisect
fi

# Determines if this commit is sycl-based or not.
function is_sycl_based_commit {
  git merge-base --is-ancestor $SYCL_ANCESTOR "$1" 2>/dev/null
}

if ! is_sycl_based_commit "$SYCL_DESCENDANT" || ! git merge-base --is-ancestor BISECT_HEAD "$SYCL_DESCENDANT"; then
  echo "SYCL_DESCENDANT not set correctly" >&2
  stop_bisect
fi

# Checks if there's a sycl-bisect-merge worktree for out-of-tree merges set up.
function has_sycl_bisect_merge_tree {
  grep "gitdir: $TOPLEVEL/.git" sycl-bisect-merge/.git &>/dev/null
}

# If BISECT_HEAD is already sycl-based, just check it out directly.
if is_sycl_based_commit BISECT_HEAD; then
  git checkout BISECT_HEAD || stop_bisect

# Otherwise, attempt a merge.
else

  # Find the next merge commit on the path to $SYCL_DESCENDANT from BISECT_HEAD.
  NEXT_MERGE=$(git rev-list --ancestry-path --merges "BISECT_HEAD..$SYCL_DESCENDANT" | tail -n 1)

  # Merges aren't used upstream, so this merge should be a sycl commit.
  is_sycl_based_commit $NEXT_MERGE || stop_bisect

  # If BISECT_HEAD is the LLVM commit being merged, check out the merge commit
  # directly.
  if [[ "$(git rev-parse BISECT_HEAD)" == "$(git rev-parse $NEXT_MERGE^2)" ]]; then
    git checkout $NEXT_MERGE || stop_bisect

  # Otherwise, create a new merge in a separate worktree and check that out.
  else

    # Switch to the worktree (if present) and check out the last sycl commit
    # that doesn't have BISECT_HEAD as an ancestor.
    BISECT_HEAD=$(git rev-parse BISECT_HEAD)
    SYCL_BISECT_MERGE_TREE=0
    if has_sycl_bisect_merge_tree; then
      SYCL_BISECT_MERGE_TREE=1
      cd sycl-bisect-merge
    else
      echo "NOTE: consider creating a sycl-bisect-merge tree to enable incremental compilation:" >&2
      echo "$ git worktree add sycl-bisect-merge" >&2
    fi
    git checkout $NEXT_MERGE^1 || stop_bisect

    # Attempt an automatic merge; back out the merge and skip this commit if it
    # fails.
    if ! git merge --quiet --no-edit $BISECT_HEAD; then
      git merge --abort
      bad_commit
    fi
    MERGED=$(git rev-parse HEAD)

    # Switch back to the main worktree and check out the new merge commit.
    if [[ $SYCL_BISECT_MERGE_TREE == 1 ]]; then
      cd ..
      git checkout $MERGED || stop_bisect
    fi
  fi
fi

# Attempt a build; skip this commit if it fails.
cmake --build build -- deploy-sycl-toolchain -j $(nproc) || bad_commit

# If a lit test is specified, run it with llvm-lit.
if [[ "$TEST" != "" ]]; then
  build/bin/llvm-lit "$TEST_WD/$TEST"
  exit
fi

# Otherwise, run the test command. If COMMAND_ALLOW_BISECT_CODES is not set,
# replace any non-zero return codes with 1.
cd "$TEST_WD"
if [[ "$COMMAND_ALLOW_BISECT_CODES" == "" ]]; then
  bash -c "$COMMAND" || exit 1
  exit
fi

# Otherwise, just run the command.
bash -c "$COMMAND"
