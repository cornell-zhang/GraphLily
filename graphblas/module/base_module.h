#ifndef __GRAPHBLAS_BASE_MODULE_H
#define __GRAPHBLAS_BASE_MODULE_H

#include "../global.h"


namespace graphblas {
namespace module {

template<typename T>
class BaseModule {
protected:
    /*! \brief The kernel name */
    std::string kernel_name_;
    /*! \brief The makefile body */
    std::string makefile_body_;

    // OpenCL runtime
    cl::Device device_;
    cl::Context context_;
    cl::Kernel kernel_;
    cl::CommandQueue command_queue_;

public:
    BaseModule(std::string kernel_name) {
        this->kernel_name_ = kernel_name;
        this->makefile_body_ = graphblas::add_kernel_to_makefile(this->kernel_name_);
        this->device_ = graphblas::find_device();
        this->context_ = cl::Context(this->device_, NULL, NULL, NULL);
    }

    /*!
     * \brief Generate the kernel header file.
     */
    void generate_kernel_header() {
        static_cast<T*>(this)->generate_kernel_header_impl();
    }

    /*!
     * \brief Link the kernel cpp file to the build directory.
     */
    void link_kernel_code();

    /*!
     * \brief Generate the Makefile.
     */
    void generate_makefile();

    /*!
     * \brief Synthesize the xclbin file. This takes a long time.
     */
    void synthesize_xclbin();

    /*!
     * \brief Send the formatted data to FPGA.
     */
    void send_data_to_FPGA() {
        static_cast<T*>(this)->send_data_to_FPGA_impl();
    }

    /*!
     * \brief Load the xclbin file and set up runtime.
     * \param xclbin_file_path The xclbin file path.
     */
    void set_up_runtime(std::string xclbin_file_path) {
        static_cast<T*>(this)->set_up_runtime_impl(xclbin_file_path);
    };
};


template <typename T>
void BaseModule<T>::link_kernel_code() {
    std::string command = "ln -s " + graphblas::root_path + "/hw/" + this->kernel_name_ + ".cpp" + " "
                          + graphblas::proj_folder_name + "/" + this->kernel_name_ + ".cpp";
    std::cout << command << std::endl;
    system(command.c_str());
    command = "ln -s " + graphblas::root_path + "/hw/" + this->kernel_name_ + ".ini" + " "
              + graphblas::proj_folder_name + "/" + this->kernel_name_ + ".ini";
    std::cout << command << std::endl;
    system(command.c_str());
}


template <typename T>
void BaseModule<T>::generate_makefile() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream makefile(graphblas::proj_folder_name + "/makefile");
    makefile << graphblas::makefile_prologue << std::endl;
    makefile << this->makefile_body_ << std::endl;
    makefile << graphblas::makefile_epilogue << std::endl;
    makefile.close();
}


template <typename T>
void BaseModule<T>::synthesize_xclbin() {
    std::string command = "mkdir -p " + graphblas::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    static_cast<T*>(this)->generate_kernel_header();
    static_cast<T*>(this)->link_kernel_code();
    static_cast<T*>(this)->generate_makefile();
    command = "cd " + graphblas::proj_folder_name + "; " + "make build";
    std::cout << command << std::endl;
    system(command.c_str());
}

} // namespace module
} // namespace graphblas

#endif // __GRAPHBLAS_BASE_MODULE_H
