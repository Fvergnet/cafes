#!/bin/bash

exec=../../build/tests/dton_two_parts
nproc=1

for i in '111'
do 
$exec \
    -stokes_ksp_type preonly \
    -stokes_pc_type lu \
    -dton_ksp_monitor_true_residual \
    -dton_ksp_rtol 1e-4 \
    -dton_ksp_type gmres \
    -dton_ksp_max_it 1000 \
    -assembling \
    -compute_singularity 0 \
    -order 2 \
    -distance 0.025 \
    -mx $i \
    -my $i \
    -saverep "Babic_extension"
done
