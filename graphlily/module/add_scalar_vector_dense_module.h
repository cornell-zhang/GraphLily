#ifndef GRAPHLILY_EWISE_ADD_MODULE_H_
#define GRAPHLILY_EWISE_ADD_MODULE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <chrono>

#include "frt.h"
#include "graphlily/global.h"
#include "graphlily/module/base_module.h"


namespace graphlily {
namespace module {

template<typename vector_data_t>
class eWiseAddModule : public BaseModule {
private:
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    const int ARG_OUT = 25, ARG_IN = 28;
public:
    /*! \brief Internal copy of the input vector */
    aligned_dense_vec_t in_;
    /*! \brief Internal copy of the output vector */
    aligned_dense_vec_t out_;

public:
    eWiseAddModule() : BaseModule() {}

    /* Overlay argument list:
    * Index       Argument                              Used in this module?
    * 0           Serpens edge list ptr                 n
    * 1-24        Serpens edge list channel             n
    * 25          Serpens vec X                         y
    * 26          Serpens vec Y                         n
    * 27          Serpens extended vec MK               n
    * 28          Serpens vec Y_out                     y
    *
    * 29          Serpens num iteration                 n
    * 30          Serpens num A mat length              n
    * 31          Serpens num rows (M)                  y
    * 32          Serpens num cols (K)                  n
    * 33          Serpens alpha (int cast)              n
    * 34          Serpens beta (int cast)               n
    * 35          value to add scalar                   y
    * 36          value to assign vector                n
    * 37          semiring zero                         n
    * 38          semiring operator                     n
    * 39          mask type                             n
    * 40          overlay kernel mode                   y
    */
    void set_unused_args() override {
        // TODO: suspend buffer transfer, but how to deal with the existed on-device memory
        // for (int i = 0; i <= 28; ++i) {
        //     if (i != 25 && i != 28) {
        //         this->instance->SuspendBuffer();
        //     }
        // }
        for (int i = 29; i <= 37; ++i) {
            if (i != 31 && i != 35) {
                this->instance->SetArg(i, 0);
            }
        }
        this->instance->SetArg(38, (char)0);
        this->instance->SetArg(39, (char)0);
        if (standalone_instance) {
            aligned_vector<int> tmp_i(1);
            aligned_vector<float> tmp_f(1);
            aligned_vector<unsigned long> tmp_ul(1);
            this->instance->SetArg(0, frtPlaceholder(tmp_i));
            for (int i = 1; i <= 24; ++i) {
                this->instance->SetArg(i, frtPlaceholder(tmp_ul));
            }
            this->instance->SetArg(26, frtPlaceholder(tmp_f));
            this->instance->SetArg(27, frtPlaceholder(tmp_f));
        }
    }

    void set_mode() override {
        this->instance->SetArg(40, (char)MODE_EWISE_ADD);
    }

    /*!
     * \brief Send the input vector from host to device.
     */
    void send_in_host_to_device(aligned_dense_vec_t &in);

    /*!
     * \brief Allocate the output buffer.
     */
    void allocate_out_buf(uint32_t len);

    /*!
     * \brief Bind the input buffer to an existing buffer.
     */
    void bind_in_buf(aligned_dense_vec_t &src_buf) {
        // this->in_ = src_buf;
    }

    /*!
     * \brief Bind the output buffer to an existing buffer.
     */
    void bind_out_buf(aligned_dense_vec_t &src_buf) {
        // this->out_ = src_buf;
    }

    /*!
     * \brief Run the module.
     * \param len The length of the in/out vector.
     * \param val The value to be added.
     */
    void run(uint32_t len, vector_data_t val);

    /*!
     * \brief Send the output vector from device to host.
     * \return The output vector.
     */
    aligned_dense_vec_t send_out_device_to_host() {
        // TODO: re-enter this function
        for (int i = 0; i <= 28; ++i) {
            if (i != ARG_OUT) this->instance->SuspendBuf(i);
        }
        this->instance->ReadFromDevice();
        this->instance->Finish();
        return this->out_;
    }

    /*!
     * \brief Compute reference results.
     * \param in The inout vector.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     * \return The output vector.
     */
    graphlily::aligned_dense_float_vec_t
    compute_reference_results(graphlily::aligned_dense_float_vec_t const &in,
                              uint32_t len,
                              float val);
};


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::send_in_host_to_device(aligned_dense_vec_t &in) {
    assert(this->standalone_instance);

    for (int i = 0; i <= 28; ++i) {
        if (i != ARG_IN) this->instance->SuspendBuf(i);
    }
    this->in_.resize(in.size());
    this->in_.assign(in.begin(), in.end());
    this->instance->SetArg(ARG_IN, frtReadWrite(this->in_));
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::allocate_out_buf(uint32_t len) {
    assert(this->standalone_instance);

    // for (int i = 0; i <= 28; ++i) {
    //     if (i != ARG_OUT) this->instance->SuspendBuf(i);
    // }
    this->out_.resize(len);
    this->instance->SetArg(ARG_OUT, frtReadWrite(this->out_));
    // this->instance->Finish();
}


template<typename vector_data_t>
void eWiseAddModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    this->set_mode();
    this->set_unused_args();

    int *val_ptr_int = (int*)(&val);
    int val_int = *val_ptr_int;

    this->instance->SetArg(31, len);
    this->instance->SetArg(35, val_int);

    this->instance->Exec();
    this->instance->Finish();
}


template<typename vector_data_t> graphlily::aligned_dense_float_vec_t
eWiseAddModule<vector_data_t>::compute_reference_results(graphlily::aligned_dense_float_vec_t const &in,
                                                         uint32_t len,
                                                         float val) {
    graphlily::aligned_dense_float_vec_t out(len);
    for (uint32_t i = 0; i < len; i++) {
        out[i] = in[i] + val;
    }
    return out;
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_EWISE_ADD_MODULE_H_
