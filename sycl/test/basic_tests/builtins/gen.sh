cat $1 | awk 'BEGIN{flag=1}/CHECK/{flag=0}flag'> 1.txt
clang++ -fpreview-breaking-changes -fsycl  -O2 $@ -S -emit-llvm -o - \
    | awk '/CLANG/{print $0}/^define/{flag=1}/^}/{flag=0;print "}\n"}flag' \
    | sed -e 's/local_unnamed.*{/{{.*}} {/' \
          -e 's/, !.*//' -e 's/) #.*/)/' \
          -e 's/%[^ ,)]*/%{{.*}}/g' \
          -e 's/metadata !.*)/metadata {{.*}})/' \
          -e 's/ *; preds =.*//' \
    | awk 'NF { printf "// CHECK-NEXT: "} !NF { printf "// CHECK-EMPTY:" } 1' \
    | sed 's/-NEXT: \([;d]\)/:      \1/' \
          >> 1.txt

mv 1.txt $1
clang-format -i $1
