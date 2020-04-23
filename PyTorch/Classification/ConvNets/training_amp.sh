#!/bin/bash
CURRENTDATE=`date +"%Y-%m-%d-%T"`
LOGFILE=run.log.amp-${CURRENTDATE}

SECONDS=0

#rocprof --obj-tracking on --hip-trace --roctx-trace -d rocout-norm \
python3.6 ./multiproc.py --nproc_per_node 2 ./main.py /data/imagenet --raport-file raport.json -j5 -p 100 \
    --data-backend pytorch \
    --arch resnext101-32x4d \
    -c fanin --label-smoothing 0.1 --workspace logs/checkpoints -b 128 --amp --static-loss-scale 128 --optimizer-batch-size 1024 --lr 2.024 --mom 0.875 --lr-schedule cosine --epochs 1 --warmup 8 --wd 6.103515625e-05 \
    --prof 100 --training-only --no-checkpoints | tee ${LOGFILE}
    #--prof 10 --training-only --no-checkpoints --enable-profiling | tee ${LOGFILE}

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed." | tee -a ${LOGFILE}
