#!/bin/bash

exec=../../build/tests/plot_singular_fields

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
    -mx 200 \
    -my 200 
