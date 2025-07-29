$exitCode = 0
$hungTests = Get-Process | Where-Object { ($_.Path -match "llvm\\install") -or ($_.Path -match "llvm\\build-e2e") -or ($_.Path -match "llvm\\build") }
$hungTests | Foreach-Object {
 $exitCode = 1
 echo "Test $($_.Path) hung!"
 tskill $_.ID
}
exit $exitCode
