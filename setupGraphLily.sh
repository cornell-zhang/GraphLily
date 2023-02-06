echo setting up Vitis-2022.1 environment......
unset LM_LICENSE_FILE
export XILINXD_LICENSE_FILE=2100@flex.ece.cornell.edu
if [ $HOSTNAME = zhang-capra-xcel.ece.cornell.edu ]; then
    source /work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/settings64.sh > /dev/null
    source /opt/xilinx/xrt/setup.sh > /dev/null
    export HLS_INCLUDE=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis_HLS/2022.1/include
    export LD_LIBRARY_PATH=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/lib/lnx64.o:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/work/shared/common/CAD_tool/Xilinx/Vitis-2022/Vitis/2022.1/lib/lnx64.o/Default:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/work/zhang-capra/common/lib:$LD_LIBRARY_PATH
    export CPATH=/usr/include/x86_64-linux-gnu:$CPATH
else
    source scl_source enable devtoolset-8
    source /opt/xilinx/2022.1/Vitis/2022.1/settings64.sh > /dev/null
    source /opt/xilinx/xrt/setup.sh > /dev/null
    export HLS_INCLUDE=/opt/xilinx/2022.1/Vitis_HLS/2022.1/include
    export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o/Default:$LD_LIBRARY_PATH
fi
echo Vitis-2022.1 setup finished
echo setting up GraphLily......
export GRAPHLILY_ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "GRAPHLILY_ROOT_PATH is set to $GRAPHLILY_ROOT_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/project_build/graphblas/software/cnpy/build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/shared/common/project_build/graphblas/software/googletest/build/lib
echo GraphLily setup finished
