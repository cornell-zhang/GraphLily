#ifndef GRAPHLILY_BASE_MODULE_H_
#define GRAPHLILY_BASE_MODULE_H_

#include "graphlily/global.h"


namespace graphlily {
namespace module {

class BaseModule {
protected:
    /*! \brief The kernel name */
    std::string kernel_name_;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    cl::Kernel kernel_;
    cl::CommandQueue command_queue_;

public:
    BaseModule(std::string kernel_name) {
        this->kernel_name_ = kernel_name;
    }

    /*!
     * \brief Get the kernel name.
     * \return The kernel name.
     */
    std::string get_kernel_name() {
        return this->kernel_name_;
    }

    /*!
     * \brief Set the device.
     */
    void set_device(cl::Device device) {
        this->device_ = device;
    }

    /*!
     * \brief Set the context.
     */
    void set_context(cl::Context context) {
        this->context_ = context;
    }

    /*!
     * \brief Set the kernel.
     */
    void set_kernel(cl::Kernel kernel) {
        this->kernel_ = kernel;
    }

    /*!
     * \brief Set the command queue.
     */
    void set_command_queue(cl::CommandQueue command_queue) {
        this->command_queue_ = command_queue;
    }

    /*!
     * \brief Set the target.
     */
    void set_target(std::string target) {
        assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
        this->target_ = target;
    }

    /*!
     * \brief Copy the contents of a buffer into another buffer without going through the host.
     */
    void copy_buffer_device_to_device(cl::Buffer src, cl::Buffer dst, size_t bytes) {
        this->command_queue_.enqueueCopyBuffer(src, dst, 0, 0, bytes);
        this->command_queue_.finish();
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
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path);
};


void BaseModule::set_up_runtime(std::string xclbin_file_path) {
    cl_int err;
    // Set this->device_ and this->context_
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    this->device_ = graphlily::find_device();
    this->context_ = cl::Context(this->device_, NULL, NULL, NULL);
    // Set this->kernel_
    auto file_buf = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    OCL_CHECK(err, this->kernel_ = cl::Kernel(program, this->kernel_name_.c_str(), &err));
    // Set this->command_queue_
    OCL_CHECK(err, this->command_queue_ = cl::CommandQueue(this->context_,
                                                           this->device_,
                                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
                                                           &err));
    // Set unused arguments
    this->set_unused_args();
    // Set the mode
    this->set_mode();
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_BASE_MODULE_H_
