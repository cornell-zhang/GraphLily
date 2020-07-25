#ifndef __GRAPHBLAS_MODULE_COLLECTION_H
#define __GRAPHBLAS_MODULE_COLLECTION_H

#include "../global.h"
#include "../module/base_module.h"


namespace graphblas {
namespace app {

using namespace module;

class ModuleCollection {
protected:
    /*! \brief The modules */
    std::vector<BaseModule*> modules_;
    /*! \brief The number of modules */
    uint32_t num_modules_ = 0;
    /*! \brief The kernel names */
    std::vector<std::string> kernel_names_;
    /*! \brief The makefile body */
    std::string makefile_body_;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    std::vector<cl::Kernel> kernels_;
    std::vector<cl::CommandQueue> command_queues_;

private:
    template <typename T> friend void module::_set_target_impl(T* t, std::string target);
    template <typename T> friend void module::_generate_makefile_impl(T* t);
    template <typename T> friend void module::_compile_impl(T* t);

public:
    /*!
     * \brief Add a module.
     * \param module The module to be added.
     */
    void add_module(BaseModule *module) {
        this->modules_.push_back(module);
        std::string kernel_name = module->get_kernel_name();
        this->kernel_names_.push_back(kernel_name);
        this->makefile_body_ += graphblas::add_kernel_to_makefile(kernel_name);
        this->num_modules_++;
    }

    /*!
     * \brief Set the target.
     */
    void set_target(std::string target) {
        _set_target_impl<ModuleCollection>(this, target);
    }

    /*!
     * \brief Generate the kernel header file.
     */
    void generate_kernel_header() {
        for (size_t i = 0; i < this->num_modules_; i++) {
            this->modules_[i]->generate_kernel_header();
        }
    }

    /*!
     * \brief Link the kernel cpp file to the proj directory.
     */
    void link_kernel_code() {
        for (size_t i = 0; i < this->num_modules_; i++) {
            this->modules_[i]->link_kernel_code();
        }
    }

    /*!
     * \brief Generate the Makefile.
     */
    void generate_makefile() {
        _generate_makefile_impl<ModuleCollection>(this);
    }

    /*!
     * \brief Compile the kernel according to this->target_.
     */
    void compile() {
        _compile_impl<ModuleCollection>(this);
    }

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path);

    /*!
     * \brief Send the formatted data to FPGA.
     */
    void send_data_to_FPGA() {
        for (size_t i = 0; i < this->num_modules_; i++) {
            this->modules_[i]->send_data_to_FPGA();
        }
    }
};


void ModuleCollection::set_up_runtime(std::string xclbin_file_path) {
    this->kernels_.resize(this->num_modules_);
    this->command_queues_.resize(this->num_modules_);
    cl_int err;
    // Set this->device_ and this->context_
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    this->device_ = graphblas::find_device();
    this->context_ = cl::Context(this->device_, NULL, NULL, NULL);
    // Set this->kernels_
    auto file_buf = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{file_buf.data(), file_buf.size()}};
    cl::Program program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }
    for (size_t i = 0; i < this->num_modules_; i++) {
        OCL_CHECK(err, this->kernels_[i] = cl::Kernel(program, this->kernel_names_[i].c_str(), &err));
    }
    // Set this->command_queues_
    for (size_t i = 0; i < this->num_modules_; i++) {
        OCL_CHECK(err, this->command_queues_[i] = cl::CommandQueue(this->context_,
                                                                   this->device_,
                                                                   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                                                   CL_QUEUE_PROFILING_ENABLE,
                                                                   &err));
    }
    // Set up runtime for each module
    for (size_t i = 0; i < this->num_modules_; i++) {
        this->modules_[i]->set_device(this->device_);
        this->modules_[i]->set_context(this->context_);
        this->modules_[i]->set_kernel(this->kernels_[i]);
        this->modules_[i]->set_command_queue(this->command_queues_[i]);
    }
}

} // namespace app
} // namespace graphblas

#endif // __GRAPHBLAS_MODULE_COLLECTION_H
