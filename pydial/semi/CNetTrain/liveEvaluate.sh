#!/bin/bash

for dataset in dstc2 Feb12 Oct11 Mar13 Mar13_incar Apr11
do
    echo "Live evaluation for ${dataset}"
    testset=${dataset}_all_test
    if [ "$dataset" == "dstc2" ]
    then
        testset=dstc2_test
    fi
    python liveDecode.py corpora/data ${testset} output/${testset}_live.json
    python corpora/scripts/score_slu.py --dataset ${testset} \
        --dataroot corpora/data --decodefile output/${testset}_live.json \
        --scorefile output/${testset}_live.score.csv \
        --ontology corpora/scripts/config/ontology_${dataset}.json \
        --trackerfile output/${testset}_live.track.json
    echo "complete."
    echo ""
done