#!/bin/bash

# Run MODE within singularity container
outdir=$1
mode PYTHON_NUMPY PYTHON_NUMPY ${outdir}/MODEConfig -outdir ${outdir}