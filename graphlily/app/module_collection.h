#ifndef GRAPHLILY_MODULE_COLLECTION_H_
#define GRAPHLILY_MODULE_COLLECTION_H_

#include "graphlily/global.h"
#include "graphlily/module/base_module.h"


namespace graphlily {
namespace app {

using namespace module;

class ModuleCollection {
protected:
    /*! \brief The modules */
    std::vector<BaseModule*> modules_;
    /*! \brief The number of modules */
    uint32_t num_modules_ = 0;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    std::vector<unsigned char> xclbin_buf_;
    cl::Program program_;

public:
    ModuleCollection() {}

    /*!
     * \brief Free up resources in the destructor.
     */
    ~ModuleCollection() {
        for (size_t i = 0; i < this->num_modules_; i++) {
            delete this->modules_[i];
        }
    }

    /*!
     * \brief Add a module.
     * \param module The module to be added.
     */
    void add_module(BaseModule *module) {
        this->modules_.push_back(module);
        this->num_modules_++;
    }

    /*!
     * \brief Set the target.
     */
    void set_target(std::string target) {
        assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
        this->target_ = target;
    }

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path);
};


void ModuleCollection::set_up_runtime(std::string xclbin_file_path) {
    // Set this->device_ and this->context_
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    this->device_ = graphlily::find_device();
    this->context_ = cl::Context(this->device_, NULL, NULL, NULL);

    // Load bitstream
    cl_int err;
    // ! maintain the binary content in `this->xclbin_buf_` to avoid destruction of xclbin buffer
    // when this function returns. Note that `cl::Program` only uses a shallow copy of binary buffer.
    this->xclbin_buf_ = xcl::read_binary_file(xclbin_file_path);
    cl::Program::Binaries binaries{{this->xclbin_buf_.data(), this->xclbin_buf_.size()}};
    this->program_ = cl::Program(this->context_, {this->device_}, binaries, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file\n";
    } else {
        std::cout << "Successfully programmed device with xclbin file\n";
    }

    // Set up runtime for each module
    for (size_t i = 0; i < this->num_modules_; i++) {
        this->modules_[i]->set_device(this->device_);
        this->modules_[i]->set_context(this->context_);
        this->modules_[i]->set_program(this->program_);

        this->modules_[i]->set_up_command_queue();
        this->modules_[i]->set_up_kernels();

        this->modules_[i]->set_unused_args();
        this->modules_[i]->set_mode();
    }
}

}  // namespace app
}  // namespace graphlily

#endif  // GRAPHLILY_MODULE_COLLECTION_H_
