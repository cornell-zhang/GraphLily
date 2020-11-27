#ifndef GRAPHLILY_APPLY_SYNTHESIZER_H_
#define GRAPHLILY_APPLY_SYNTHESIZER_H_

#include "graphlily/synthesizer/base_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class ApplySynthesizer : public BaseSynthesizer {
public:
    ApplySynthesizer() : BaseSynthesizer("kernel_apply") {}

    void generate_kernel_header() override {
        // No need to modify the header file since there are no configuration parameters
    }

    void generate_kernel_ini() override {
        // No need to modify the ini file since there are no configuration parameters
    }
};

}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_APPLY_SYNTHESIZER_H_
