#ifndef __GRAPHBLAS_BASE_MODULE_H
#define __GRAPHBLAS_BASE_MODULE_H

#include "../global.h"


namespace graphblas {
namespace module {

template <typename T>
void _set_target_impl(T* t, std::string target) {
    assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
    t->target_ = target;
}


template <typename T>
void _generate_makefile_impl(T* t) {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream makefile(graphblas::proj_folder_name + "/makefile");
    makefile << "TARGET := " << t->target_ << "\n" << std::endl;
    makefile << graphblas::makefile_prologue << t->makefile_body_ << graphblas::makefile_epilogue;
    makefile.close();
}


template <typename T>
void _compile_impl(T* t) {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    t->generate_kernel_header();
    t->generate_kernel_ini();
    t->link_kernel_code();
    t->generate_makefile();
    command = "cd " + graphblas::proj_folder_name + "; " + "make build";
    std::cout << command << std::endl;
    system(command.c_str());
    if (t->target_ == "sw_emu" || t->target_ == "hw_emu") {
        command = "cp " + graphblas::proj_folder_name + "/emconfig.json " + ".";
        std::cout << command << std::endl;
        system(command.c_str());
    }
}


class BaseModule {
protected:
    /*! \brief The kernel name */
    std::string kernel_name_;
    /*! \brief The makefile body */
    std::string makefile_body_;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    cl::Kernel kernel_;
    cl::CommandQueue command_queue_;

private:
    template <typename T> friend void _set_target_impl(T* t, std::string target);
    template <typename T> friend void _generate_makefile_impl(T* t);
    template <typename T> friend void _compile_impl(T* t);

public:
    BaseModule(std::string kernel_name) {
        this->kernel_name_ = kernel_name;
        this->makefile_body_ = graphblas::add_kernel_to_makefile(this->kernel_name_);
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
        _set_target_impl<BaseModule>(this, target);
    }

    /*!
     * \brief Copy the contents of a buffer into another buffer without going through the host.
     */
    void copy_buffer_device_to_device(cl::Buffer src, cl::Buffer dst, size_t bytes) {
        this->command_queue_.enqueueCopyBuffer(src, dst, 0, 0, bytes);
        this->command_queue_.finish();
    }

    /*!
     * \brief Generate the kernel header file.
     */
    virtual void generate_kernel_header() = 0;

    /*!
     * \brief Generate the kernel .ini configuration file.
     */
    virtual void generate_kernel_ini() = 0;

    /*!
     * \brief Link the kernel cpp file to the proj directory.
     */
    virtual void link_kernel_code();

    /*!
     * \brief Generate the Makefile.
     */
    virtual void generate_makefile() {
        _generate_makefile_impl<BaseModule>(this);
    }

    /*!
     * \brief Compile the kernel according to this->target_.
     */
    virtual void compile() {
         _compile_impl<BaseModule>(this);
    }

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path);
};


void BaseModule::link_kernel_code() {
    std::string command = "ln -s " + graphblas::root_path + "/graphblas/hw/" + this->kernel_name_ + ".cpp"
                        + " " + graphblas::proj_folder_name + "/" + this->kernel_name_ + ".cpp";
    std::cout << command << std::endl;
    system(command.c_str());
}


void BaseModule::set_up_runtime(std::string xclbin_file_path) {
    cl_int err;
    // Set this->device_ and this->context_
    if (this->target_ == "sw_emu" || this->target_ == "hw_emu") {
        setenv("XCL_EMULATION_MODE", this->target_.c_str(), true);
    }
    this->device_ = graphblas::find_device();
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
}

} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_BASE_MODULE_H
