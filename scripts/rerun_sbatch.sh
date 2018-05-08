#!/bin/bash
rm out/sbatch.*
sbatch run_sbatch.sh $1
sleep 2
tail -f out/sbatch.*
