#!/bin/bash
set -eux
for e in Humanoid-v2 #HalfCheetah-v2 #Ant-v2 # Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python run_expert.py ./experts/$e.pkl $e --render --simulate 2
done
