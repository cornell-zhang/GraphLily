#ifndef GRAPHLILY_SYNTHESIZER_COLLECTION_H_
#define GRAPHLILY_SYNTHESIZER_COLLECTION_H_

#include "graphlily/global.h"
#include "graphlily/synthesizer/base_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class SynthesizerCollection {
protected:
    /*! \brief The synthesizers */
    std::vector<BaseSynthesizer*> synthesizers_;
    /*! \brief The number of synthesizers */
    uint32_t num_synthesizers_ = 0;
    /*! \brief The makefile body */
    std::string makefile_body_;
    /*! \brief The target; can be sw_emu, hw_emu, hw */
    std::string target_;

private:
    template<typename T> friend void synthesizer::_generate_makefile_impl(T* t);
    template<typename T> friend void synthesizer::_synthesize_impl(T* t);

public:
    /*!
     * \brief Add a synthesizer.
     * \param synthesizer The synthesizer to be added.
     */
    void add_synthesizer(BaseSynthesizer *synthesizer) {
        this->synthesizers_.push_back(synthesizer);
        this->makefile_body_ += graphlily::add_kernel_to_makefile(synthesizer->get_kernel_name());
        this->num_synthesizers_++;
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
    void generate_kernel_header() {
        for (size_t i = 0; i < this->num_synthesizers_; i++) {
            this->synthesizers_[i]->generate_kernel_header();
        }
    }

    /*!
     * \brief Generate the kernel .ini configuration file.
     */
    void generate_kernel_ini() {
        for (size_t i = 0; i < this->num_synthesizers_; i++) {
            this->synthesizers_[i]->generate_kernel_ini();
        }
    }

    /*!
     * \brief Link the kernel cpp file to the proj directory.
     */
    void link_kernel_code() {
        for (size_t i = 0; i < this->num_synthesizers_; i++) {
            this->synthesizers_[i]->link_kernel_code();
        }
    }

    /*!
     * \brief Generate the Makefile.
     */
    void generate_makefile() {
        _generate_makefile_impl<SynthesizerCollection>(this);
    }

    /*!
     * \brief Compile the kernel according to this->target_.
     */
    void synthesize() {
        _synthesize_impl<SynthesizerCollection>(this);
    }
};

}  // namespace synthesizer
}  // namespace graphlily

#endif  // #define GRAPHLILY_SYNTHESIZER_COLLECTION_H_
