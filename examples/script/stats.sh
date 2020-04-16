#!/usr/bin/env bash

echo ==========
echo SMAC single
perl ./script/find_best.pl $(fd hists smac-single)

echo ==========
echo ADDTREE single
perl ./script/find_best.pl $(fd json addtree-single)

echo ==========
echo RANDOM single
perl ./script/find_best.pl $(fd txt random-single)

echo =============================================
echo ADDTREE multiple
perl ./script/find_best.pl $(fd json addtree-multiple)

echo ==========
echo RANDOM multiple
perl ./script/find_best.pl $(fd txt random-multiple) $(fd json random-multiple)

echo ==========
echo TPE multiple
perl ./script/find_best.pl $(fd hists tpe-multiple)

echo =============================================
echo "ADDTREE multiple (new results)"
cd addtree-multiple-new
for expdir in `fd -t d`; do
    expid=${expdir: -6}
    if [ -f $expdir/addtree_iter_1.json ]; then
        echo -n $expid:
        perl ../script/find_best.pl `fd json $expdir`
    fi
done
cd -

echo ==========
echo "RANDOM multiple (new results)"
cd random-multiple-new
for expdir in `fd -t d`; do
    expid=${expdir: -6}
    if [ -f $expdir/random_iter_1.json ]; then
        echo -n $expid:
        perl ../script/find_best.pl `fd json $expdir`
    fi
done
cd -


echo ==========
echo "TPE multiple (new results)"
cd tpe-multiple-new
for expdir in `fd -t d`; do
    expid=${expdir: -6}
    if [ -f $expdir/tpe_iter_hists.txt ]; then
        echo -n $expid:
        perl ../script/find_best.pl $expdir/tpe_iter_hists.txt
    fi
done
cd -
