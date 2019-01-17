#!/bin/bash

export CAFFE_ROOT=/root/src/caffe/build

cd /home/bharadwaj/ImageSegmentation/data/streetObjects2Data/

logfile="logFile/trainlogs/log-$(date +%F_%R).log"
#logfile="logFile/testlogs/$1_log-$(date +%F_%R).log"
##########################################################################################################################################################################
#snapshotIter="$(ls snapshot_noDepth/*.solverstate.h5 | grep -o "_[0-9]*[.]" | sed -e "s/^_//" -e "s/.$//" | xargs printf "%010d\n" | sort | tail -1 | sed -e "s/^0*//")"

#if [ "x${snapshotIter}" != "x" ]; then
#lastSnapshot="snapshot_noDepth/snapshot_iter_${snapshotIter}.solverstate.h5"
#    continueFromSnapshot="-snapshot ${lastSnapshot}"
#    echo "Continuing from snapshot ${lastSnapshot}" > ${logfile}
#else
#    continueFromSnapshot=""
#    echo "Starting new training" > ${logfile}
#fi 
##########################################################################################################################################################################

#${CAFFE_ROOT}/tools/caffe train   --solver=solver.prototxt -snapshot=snapshots/snapshot_iter_10000.solverstate 2>&1| tee ${logfile}
${CAFFE_ROOT}/tools/caffe train   --solver=solver.prototxt  2>&1| tee ${logfile} 
#${CAFFE_ROOT}/tools/caffe test  -model resnet34.prototxt -weights snapshots/$1 -gpu 0 -iterations 669 2>&1| tee ${logfile}
#${CAFFE_ROOT}/tools/caffe test  -model fcn8.prototxt -weights snapshots/$1 -gpu 0 -iterations 36 2>&1| tee ${logfile} 
