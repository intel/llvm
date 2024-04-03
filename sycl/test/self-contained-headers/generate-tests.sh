
if [[ $# < 1 ]]; then
  echo "First argument must point to a folder with sycl headers"
  exit 1
fi

output_dir=$(dirname "$0")

if [[ $# > 1 ]]; then
  output_dir=$2
  echo "Output files will be written to $output_dir"
else
  echo "Output files will be written to $output_dir"
  echo "You can override this by passing the second argument to the script"
fi

echo "Cleaning up old .cpp files in $output_dir"
for file in $(find $output_dir -name "*.cpp"); do
  rm $file
done

echo "Generating new .cpp files"
for header in $(ls $1); do
  path="$1/$header"
  if [[ -d $path ]]; then
    continue
  fi

  if [[ "$header" != *".hpp" ]]; then
    continue;
  fi

  filename="${header%.hpp}.cpp"
  output_file="$output_dir/$filename"

  echo "// RUN: %clangxx -fsycl -fsyntax-only %s" > $output_file
  echo "#include <sycl/$header>" >> $output_file
done

echo "Done"
