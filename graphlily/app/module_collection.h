#ifndef GRAPHLILY_MODULE_COLLECTION_H_
#define GRAPHLILY_MODULE_COLLECTION_H_

#include "frt.h"
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

    fpga::Instance *instance = nullptr;

public:
    ModuleCollection() {}

    /*!
     * \brief Free up resources in the destructor.
     */
    ~ModuleCollection() {
        for (size_t i = 0; i < this->num_modules_; i++) {
            delete this->modules_[i];
        }
        if (this->instance != nullptr) delete this->instance;
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
    void set_up_runtime(std::string xclbin_file_path) {
        this->instance = new fpga::Instance(xclbin_file_path);
        // Set up runtime for each module
        for (size_t i = 0; i < this->num_modules_; i++) {
            this->modules_[i]->set_up_runtime(this->instance);
        }
    }
};

}  // namespace app
}  // namespace graphlily

#endif  // GRAPHLILY_MODULE_COLLECTION_H_
