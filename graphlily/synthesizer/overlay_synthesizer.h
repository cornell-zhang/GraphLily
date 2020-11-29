#ifndef GRAPHLILY_OVERLAY_SYNTHESIZER_H_
#define GRAPHLILY_OVERLAY_SYNTHESIZER_H_

#include "graphlily/synthesizer/synthesizer_collection.h"
#include "graphlily/synthesizer/spmv_spmspv_synthesizer.h"
#include "graphlily/synthesizer/apply_synthesizer.h"


namespace graphlily {
namespace synthesizer {

class OverlaySynthesizer : public SynthesizerCollection {
private:
    graphlily::synthesizer::SpmvSpmspvSynthesizer* spmv_spmspv_synthesizer_;
    graphlily::synthesizer::ApplySynthesizer* apply_synthesizer_;

public:
    OverlaySynthesizer(uint32_t num_channels, uint32_t out_buf_len, uint32_t vec_buf_len) {
        this->spmv_spmspv_synthesizer_ = new graphlily::synthesizer::SpmvSpmspvSynthesizer(
            num_channels,
            out_buf_len,
            vec_buf_len);
        this->add_synthesizer(this->spmv_spmspv_synthesizer_);

        this->apply_synthesizer_ = new graphlily::synthesizer::ApplySynthesizer();
        this->add_synthesizer(this->apply_synthesizer_);
    }
};

}  // namespace synthesizer
}  // namespace graphlily

#endif  // GRAPHLILY_OVERLAY_SYNTHESIZER_H_
