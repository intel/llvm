# Contributing to SYCL-MLIR

## 1) Fork the Intel LLVM repository

We recommend contributors to work in their own fork of the Intel LLVM project. To create a fork:
	
  1) Visit https://github.com/intel/llvm and click the Fork button to establish a fork (e.g. intel-llvm)
  2) Go to your newly created fork, and get the URL of you fork (green Code button):
	
  ```sh
	  https://github.com/<user>/intel-llvm.git    
	  where <user> is your GitHub username.	
  ```

## 2) Setup the forked repository

Create your clone of the sycl-mlir branch (\<user\> stands in for your GitHub username):

  ```sh
  # Clone the forked repository, setup upstream repo.
  git clone --recursive https://github.com/<user>/intel-llvm.git -b sycl-mlir
  cd intel-llvm
  git remote add upstream https://github.com/intel/llvm.git
  # Ensure pushing to the upstream repository is not allowed.
  git remote set-url --push upstream no_push
  
  # Verify that your remote is setup correctly, the command output should be:
  # origin  https://github.com/<user>/intel-llvm.git (fetch)
  # origin  https://github.com/<user>/intel-llvm.git (push)
  # upstream  https://github.com/intel/llvm.git (fetch)
  # upstream  no_push (push)
  git remote -v
  ```

After the commands above, *upstream* refers to the original Intel LLVM repository while *origin* refers to your own fork of the repository.

IMPORTANT: You should be careful to not commit changes to your fork's *sycl-mlir* branch to avoid getting out of sync with the *sycl-mlir* branch in the *upstream* repository. 

## 3) Keeping your fork up to date 

To update your fork, first fetch the latest version of both your fork and the upstream repositories: 

```sh 
git fetch --all
```

Then update your fork *sycl-mlir* branch:

```sh
git checkout sycl-mlir
git merge origin/sycl-mlir
git merge upstream/sycl-mlir
git push origin sycl-mlir
```

Finally ensure that the *sycl-mlir* branch in your fork (origin) and the upstream one are identical (this should be the case if you never committed code into your fork's *sycl-mlir* branch directly):

```sh
git diff upstream/sycl-mlir
```

The command should report no differences.

## 4) Build and Testing

Create a branch in your local fork:

```sh
git checkout sycl-mlir
git checkout -b <name-of-your-branch> 
```

Configure and build the project:

```sh
python ./buildbot/configure.py --build-type Debug   
python ./buildbot/compile.py   
```

You can now make local changes in your newly created branch. Once you have completed the changes please ensure all relevant lit tests pass:

```sh
ninja check-mlir-sycl 
ninja check-cgeist
ninja check polygeist-opt
```

## 5) Commit changes

At this point you are ready to commit and push your changes. Although not mandatory, is good practive to sign your commits (git commit -s) before pushing them. Also please prepend commit messages with `[SYCL-MLIR]` for easy identification:

```sh
git commit -s -m "[SYCL-MLIR]: Meaningful msg"
# push the commit to your local branch
git push origin <name-of-your-branch> 
``` 

Note: Additional tags can be used to identify the component affected by the change (e.g. [SYCLToLLVM] would identify the SYCL to LLVM conversion pass).

## 6) Pull Request

Pull request should be created on the upstream repository. By pushing changes to your fork (`git push origin <name-of-your-branch>` as described in section 5, the upstream repository is 'aware' of the changes. Go to https://github.com/intel/llvm and create a PR against the base branch (`sycl-mlir`). 

IMPORTANT: Ensure PRs are against the `sycl-mlir` branch, the default is the `sycl` branch! Also please tag your PR with the label 'sycl-mlir'.
