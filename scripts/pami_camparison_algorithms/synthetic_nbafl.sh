#!/bin/sh
ps -ef | grep python | cut -c 9-15 | xargs kill -s 9
python main.py --cfg /home/pkl2/lyn/FederatedScope/scripts/pami_camparison_algorithms/asyn_dp_synthetic_nbafl.yaml