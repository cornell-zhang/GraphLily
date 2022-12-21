#! /bin/bash

WORK_DIR=run
mkdir -p "${WORK_DIR}"

tapac \
  --work-dir "${WORK_DIR}" \
  --top Serpens \
  --platform xilinx_u280_gen3x16_xdma_1_202211_1 \
  --clock-period 3.33 \
  --read-only-args edge_list_ptr \
  --read-only-args edge_list_ch* \
  --read-only-args vec_Y \
  --run-floorplan-dse \
  --enable-synth-util \
  --enable-hbm-binding-adjustment \
  --max-parallel-synth-jobs 32 \
  -o "${WORK_DIR}/Serpens.xo" \
  --floorplan-output "${WORK_DIR}/Serpens_floorplan.tcl" \
  --connectivity ./link_config.ini \
  ./graphlily.cpp \
  2>&1 | tee tapac.log
