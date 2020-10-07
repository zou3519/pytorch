#!/bin/bash
set -x

today=$(date +%Y%m%d%H%M%S)
foldername=batched_grad_$today
outdir=~/profout/$foldername
mkdir $outdir

check_mode () {
  mode=$1
  file=$outdir/test_autograd_$mode
  GRADCHECK_MODE=$mode pytest test/test_autograd.py --tb='short' -v --result-log=$file.txt --json-report-file=$file.json --json-report
  python analysis/construct_table.py --type=$mode --date=$today --logfile $file.json >> $outdir/out.csv
  python analysis/gather_warnings.py --logfile $file.json >> $outdir/missing_rules.csv

  file=$outdir/test_nn_$mode
  GRADCHECK_MODE=$mode pytest test/test_nn.py --tb='short' -v --result-log=$file.txt --json-report-file=$file.json --json-report
  python analysis/construct_table.py --type=$mode --date=$today --logfile $file.json >> $outdir/out.csv
  python analysis/gather_warnings.py --logfile $file.json >> $outdir/missing_rules.csv
}

check_mode jacobian
check_mode hessian
check_mode doublebatched
