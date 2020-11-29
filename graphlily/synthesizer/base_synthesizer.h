#ifndef GRAPHLILY_BASE_SYNTHESIZER_H_
#define GRAPHLILY_BASE_SYNTHESIZER_H_

#include "graphlily/global.h"


namespace graphlily {
namespace synthesizer {

template<typename T>
void _generate_makefile_impl(T* t) {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    std::ofstream makefile(graphlily::proj_folder_name + "/makefile");
    makefile << "TARGET := " << t->target_ << "\n" << std::endl;
    makefile << graphlily::makefile_prologue << t->makefile_body_ << graphlily::makefile_epilogue;
    makefile.close();
}


template<typename T>
void _synthesize_impl(T* t) {
    std::string command = "mkdir -p " + graphlily::proj_folder_name;
    std::cout << command << std::endl;
    system(command.c_str());
    t->link_kernel_code();
    t->generate_kernel_header();
    t->generate_kernel_ini();
    t->generate_makefile();
    command = "cd " + graphlily::proj_folder_name + "; " + "make build";
    std::cout << command << std::endl;
    system(command.c_str());
    if (t->target_ == "sw_emu" || t->target_ == "hw_emu") {
        command = "cp " + graphlily::proj_folder_name + "/emconfig.json " + ".";
        std::cout << command << std::endl;
        system(command.c_str());
    }
}


class BaseSynthesizer {
protected:
    /*! \brief The kernel name */
    std::string kernel_name_;
    /*! \brief The makefile body */
    std::string makefile_body_;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

private:
    template<typename T> friend void _generate_makefile_impl(T* t);
    template<typename T> friend void _synthesize_impl(T* t);

public:
    BaseSynthesizer(std::string kernel_name) {
        this->kernel_name_ = kernel_name;
        this->makefile_body_ = graphlily::add_kernel_to_makefile(this->kernel_name_);
    }

    /*!
     * \brief Get the kernel name.
     * \return The kernel name.
     */
    std::string get_kernel_name() {
        return this->kernel_name_;
    }

    /*!
     * \brief Set the target.
     */
    void set_target(std::string target) {
       assert(target == "sw_emu" || target == "hw_emu" || target == "hw");
       this->target_ = target;
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
        _generate_makefile_impl<BaseSynthesizer>(this);
    }

    /*!
     * \brief Synthesize the kernel according to this->target_.
     */
    virtual void synthesize() {
         _synthesize_impl<BaseSynthesizer>(this);
    }
};


void BaseSynthesizer::link_kernel_code() {
    std::string command = "cp " + graphlily::root_path + "/graphlily/hw/" + "*.h"
                                + " " + graphlily::proj_folder_name + "/";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + this->kernel_name_ + ".cpp"
                    + " " + graphlily::proj_folder_name + "/" + this->kernel_name_ + ".cpp";
    std::cout << command << std::endl;
    system(command.c_str());

    command = "cp " + graphlily::root_path + "/graphlily/hw/" + this->kernel_name_ + ".ini"
                    + " " + graphlily::proj_folder_name + "/" + this->kernel_name_ + ".ini";
    std::cout << command << std::endl;
    system(command.c_str());
}

}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_BASE_SYNTHESIZER_H_
