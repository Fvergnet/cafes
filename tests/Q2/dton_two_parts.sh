#!/bin/bash

exec=../../build/tests/dton_two_parts
nproc=1

for i in '61'
do
$exec \
    -stokes_ksp_type preonly \
    -stokes_pc_type lu \
    -dton_ksp_monitor_true_residual \
    -dton_ksp_rtol 1e-5 \
    -dton_ksp_type gmres \
    -assembling \
    -compute_singularity 1\
    -order 2 \
    -distance 0.025 \
    -mx $i \
    -my $i \
    -saverep "extension_with_chix"
done
