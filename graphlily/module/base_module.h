#ifndef GRAPHLILY_BASE_MODULE_H_
#define GRAPHLILY_BASE_MODULE_H_

#include "frt.h"
#include "graphlily/global.h"


namespace graphlily {
namespace module {

class BaseModule {
protected:
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    fpga::Instance *instance = nullptr;
    bool standalone_instance = false;

public:
    BaseModule() {}

    ~BaseModule() {
        if (this->standalone_instance) delete this->instance;
    }

    /*!
     * \brief Set the target.
     */
    void set_target(std::string target) {
        assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
        this->target_ = target;
    }

    /*!
     * \brief Set unused arguments
     */
    virtual void set_unused_args() = 0;

    /*!
     * \brief Set the mode. SpMV and SpMSpV are merged into a single kernel; we need to select
     *        one of them, so called the mode. Similarly, all apply functions are merged into one kernel.
     */
    virtual void set_mode() = 0;

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param runtime The instance of xclbin as fpga::Instance.
     */
    void set_up_runtime(fpga::Instance *runtime) {
        this->instance = runtime;
    }

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path) {
        this->instance = new fpga::Instance(xclbin_file_path);
        this->standalone_instance = true;
    }
};

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_BASE_MODULE_H_
