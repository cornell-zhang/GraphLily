echo setting up Vitis-2022.1 environment......
unset LM_LICENSE_FILE
source scl_source enable devtoolset-8
export XILINXD_LICENSE_FILE=2100@flex.ece.cornell.edu
if [ $HOSTNAME = brg-zhang-xcel.ece.cornell.edu ]; then
    module load xilinx-vivado-vitis_2022.1.2
    export HLS_INCLUDE=/opt/xilinx/2022.1/Vitis_HLS/2022.1/include
elif [ $HOSTNAME = zhang-capra-xcel.ece.cornell.edu ]; then
    source /work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
    export HLS_INCLUDE=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis_HLS/2022.1/include
    export LD_LIBRARY_PATH=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/lib/lnx64.o:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/lib/lnx64.o/Default:$LD_LIBRARY_PATH
else
    source /opt/xilinx/2022.1/Vitis/2022.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
    export HLS_INCLUDE=/opt/xilinx/2022.1/Vitis_HLS/2022.1/include
    export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o/Default:$LD_LIBRARY_PATH
fi
echo Vitis-2022.1 setup finished
