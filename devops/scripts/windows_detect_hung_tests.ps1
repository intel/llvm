$exitCode = 0
$hungTests = Get-Process | Where-Object { ($_.Path -match "llvm\\install") -or ($_.Path -match "llvm\\build-e2e") }
$hungTests | Foreach-Object {
 $exitCode = 1
 echo "Test $($_.Path) hung!"
 Stop-Process -Force $_
}
exit $exitCode
