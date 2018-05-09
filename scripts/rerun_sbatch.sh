#!/bin/bash
rm out/sbatch.*
sbatch run_sbatch.sh $1
sleep 5
tail -f out/sbatch.*
