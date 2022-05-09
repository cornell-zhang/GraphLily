#ifndef GRAPHLILY_BASE_MODULE_H_
#define GRAPHLILY_BASE_MODULE_H_

#include "graphlily/global.h"


namespace graphlily {
namespace module {

class BaseModule {
protected:
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    // OpenCL runtime shared across overlay platform
    cl::Device device_;
    cl::Context context_;
    cl::Program program_;

    // ! only used by `set_up_split_kernel_runtime`, to maintain xclbin content
    // for `program_`, since `cl::Program` uses a shallow copy of binary buffer.
    std::vector<unsigned char> xclbin_buf_;

    // Separated command queue to enable inter-kernel concurrency. Note that ONE
    // shared command queue with `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` open
    // also works to allow running multiple kernels concurrently.
    cl::CommandQueue command_queue_;

    // TODO: simple workaround for split-kernel
    // TODO: need to find a way to organize it
    cl::Kernel spmspv_apply_;
    cl::Kernel spmv_result_drain_;
    cl::Kernel spmv_sk0_;
    cl::Kernel spmv_sk1_;
    cl::Kernel spmv_sk2_;
    cl::Kernel spmv_vector_loader_;

public:
    BaseModule() {}

    virtual ~BaseModule() {
        this->device_ = nullptr;
        this->context_ = nullptr;
        this->program_ = nullptr;
        // ! `this->command_queue_` only construct & destruct inside this class
        // this->command_queue_ = nullptr;
    }

    // /*!
    //  * \brief Get the kernel name.
    //  * \return The kernel name.
    //  */
    // std::string get_kernel_name() {
    //     return this->kernel_name_;
    // }

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
     * \brief Set the program.
     */
    void set_program(cl::Program program) {
        this->program_ = program;
    }

    /*!
     * \brief Set up the split kernels from scratch.
     */
    void set_up_kernels();

    // /*!
    //  * \brief Set the kernel.
    //  */
    // void set_kernel(cl::Kernel kernel) {
    //     this->kernel_ = kernel;
    // }

    // /*!
    //  * \brief Set the command queue to an existed instance.
    //  */
    // void set_command_queue(cl::CommandQueue command_queue) {
    //     this->command_queue_ = command_queue;
    // }

    /*!
     * \brief Set up the command queue from scratch.
     */
    void set_up_command_queue();

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
     * \brief Load the xclbin file and set up runtime,
              alias of `set_up_split_kernel_runtime`.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path);

    /*!
     * \brief Load the xclbin file and set up runtime
     *        for the design using split kernels.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_split_kernel_runtime(std::string xclbin_file_path);
};

void BaseModule::set_up_kernels() {
    cl_int err;
    cl::Program &program = this->program_;
    OCL_CHECK(err, this->spmspv_apply_ = cl::Kernel(program, "spmspv_apply", &err));
    OCL_CHECK(err, this->spmv_sk0_ = cl::Kernel(program, "spmv_sk0", &err));
    OCL_CHECK(err, this->spmv_sk1_ = cl::Kernel(program, "spmv_sk1", &err));
    OCL_CHECK(err, this->spmv_sk2_ = cl::Kernel(program, "spmv_sk2", &err));
    OCL_CHECK(err, this->spmv_vector_loader_ = cl::Kernel(program, "spmv_vector_loader", &err));
    OCL_CHECK(err, this->spmv_result_drain_ = cl::Kernel(program, "spmv_result_drain", &err));
}

void BaseModule::set_up_command_queue() {
    cl_int err;
    OCL_CHECK(err, this->command_queue_ =
                        cl::CommandQueue(this->context_, this->device_,
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                            CL_QUEUE_PROFILING_ENABLE,
                                        &err));
}

void BaseModule::set_up_split_kernel_runtime(std::string xclbin_file_path) {
    // Set this->device_ and this->context_
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    this->device_ = graphlily::find_device();
    this->context_ = cl::Context(this->device_, NULL, NULL, NULL);

    // Load bitstream
    cl_int err;
    this->xclbin_buf_ = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{this->xclbin_buf_.data(), this->xclbin_buf_.size()}};
    this->program_ = cl::Program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }

    // Set up command queue from this->device_ and this->context_
    set_up_command_queue();
    // Set up kernels from this->program_
    set_up_kernels();

    // Set unused arguments
    set_unused_args();
    // Set the overlay mode
    set_mode();
}

void BaseModule::set_up_runtime(std::string xclbin_file_path) {
    // alias of `set_up_split_kernel_runtime`
    this->set_up_split_kernel_runtime(xclbin_file_path);
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_BASE_MODULE_H_
