#ifndef GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_
#define GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_

#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>

#include "graphlily/global.h"
#include "graphlily/module/base_module.h"


namespace graphlily {
namespace module {

template<typename vector_data_t>
class AssignVectorDenseModule : public BaseModule {
private:
    using packed_val_t = struct {vector_data_t data[graphlily::pack_size];};
    using aligned_dense_vec_t = std::vector<vector_data_t, aligned_allocator<vector_data_t>>;

    /*! \brief The mask type */
    graphlily::MaskType mask_type_;

    const int ARG_MASK = 25, ARG_INOUT = 27;

public:
    /*! \brief Internal copy of mask */
    aligned_dense_vec_t mask_;
    /*! \brief Internal copy of inout */
    aligned_dense_vec_t inout_;

public:
    AssignVectorDenseModule() : BaseModule() {}

    /* Overlay argument list:
    * Index       Argument                              Used in this module?
    * 0           Serpens edge list ptr                 n
    * 1-24        Serpens edge list channel             n
    * 25          Serpens vec X                         y
    * 26          Serpens vec Y                         n
    * 27          Serpens extended vec MK               y
    * 28          Serpens vec Y_out                     n
    *
    * 29          Serpens num iteration                 n
    * 30          Serpens num A mat length              n
    * 31          Serpens num rows (M)                  y
    * 32          Serpens num cols (K)                  y
    * 33          Serpens alpha (int cast)              n
    * 34          Serpens beta (int cast)               n
    * 35          value to add scalar                   n
    * 36          value to assign vector                y
    * 37          semiring zero                         n
    * 38          semiring operator                     n
    * 39          mask type                             y
    * 40          overlay kernel mode                   y
    */
    void set_unused_args() override {
        // TODO: suspend buffer transfer, but how to deal with the existed on-device memory
        // for (int i = 0; i <= 28; ++i) {
        //     if (i != 25 && i != 27) {
        //         this->instance->SuspendBuffer();
        //     }
        // }
        for (int i = 29; i <= 37; ++i) {
            if (i != 31 && i != 32 && i != 36) {
                this->instance->SetArg(i, 0);
            }
        }
        this->instance->SetArg(38, (char)0);
        if (standalone_instance) {
            aligned_vector<int> tmp_i(1);
            aligned_vector<float> tmp_f(1);
            aligned_vector<unsigned long> tmp_ul(1);
            this->instance->SetArg(0, frtPlaceholder(tmp_i));
            for (int i = 1; i <= 24; ++i) {
                this->instance->SetArg(i, frtPlaceholder(tmp_ul));
            }
            this->instance->SetArg(26, frtPlaceholder(tmp_f));
            this->instance->SetArg(28, frtPlaceholder(tmp_f));
        }
    }

    void set_mode() override {
        this->instance->SetArg(40, (char)MODE_DENSE_ASSIGN);
    }

    /*!
     * \brief Set the mask type.
     * \param mask_type The mask type.
     */
    void set_mask_type(graphlily::MaskType mask_type) {
        if (mask_type == graphlily::kNoMask) {
            std::cerr << "Please set the mask type" << std::endl;
            exit(EXIT_FAILURE);
        } else {
            this->mask_type_ = mask_type;
        }
    }

    /*!
     * \brief Send the mask from host to device.
     */
    void send_mask_host_to_device(aligned_dense_vec_t &mask);

    /*!
     * \brief Send the inout from host to device.
     */
    void send_inout_host_to_device(aligned_dense_vec_t &inout);

    /*!
     * \brief Bind the mask buffer to an existing buffer.
     */
    void bind_mask_buf(aligned_dense_vec_t &src_buf) {
        // this->mask_ = src_buf;
    }

    /*!
     * \brief Bind the inout buffer to an existing buffer.
     */
    void bind_inout_buf(aligned_dense_vec_t &src_buf) {
        // this->inout_ = src_buf;
    }

    /*!
     * \brief Run the module.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void run(uint32_t len, vector_data_t val);

    /*!
     * \brief Send the mask from device to host.
     * \return The mask.
     */
    aligned_dense_vec_t send_mask_device_to_host() {
        return this->mask_;
    }

    /*!
     * \brief Send the inout from device to host.
     * \return The inout.
     */
    aligned_dense_vec_t send_inout_device_to_host() {
        // TODO: re-enter this function
        for (int i = 0; i <= 28; ++i) {
            if (i != ARG_INOUT) this->instance->SuspendBuf(i);
        }
        this->instance->ReadFromDevice();
        this->instance->Finish();
        return this->inout_;
    }

    /*!
     * \brief Compute reference results.
     * \param mask The mask vector.
     * \param inout The inout vector.
     * \param len The length of the mask/inout vector.
     * \param val The value to be assigned to the inout vector.
     */
    void compute_reference_results(graphlily::aligned_dense_float_vec_t &mask,
                                   graphlily::aligned_dense_float_vec_t &inout,
                                   uint32_t len,
                                   float val);
};


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_mask_host_to_device(aligned_dense_vec_t &mask) {
    assert(this->standalone_instance);

    for (int i = 0; i <= 28; ++i) {
        if (i != ARG_MASK) this->instance->SuspendBuf(i);
    }
    this->mask_.assign(mask.begin(), mask.end());
    this->instance->SetArg(ARG_MASK, frtReadWrite(this->mask_));
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::send_inout_host_to_device(aligned_dense_vec_t &inout) {
    assert(this->standalone_instance);

    for (int i = 0; i <= 28; ++i) {
        if (i != ARG_INOUT) this->instance->SuspendBuf(i);
    }
    this->inout_.assign(inout.begin(), inout.end());
    this->instance->SetArg(ARG_INOUT, frtReadWrite(this->inout_));
    this->instance->WriteToDevice();
    // this->instance->Finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::run(uint32_t len, vector_data_t val) {
    this->set_mode();
    this->set_unused_args();

    int *val_ptr_int = (int*)(&val);
    int val_int = *val_ptr_int;

    this->instance->SetArg(31, len);
    this->instance->SetArg(32, len); // read num of len from vec_X
    this->instance->SetArg(36, val_int);
    this->instance->SetArg(39, (char)this->mask_type_);

    this->instance->Exec();
    this->instance->Finish();
}


template<typename vector_data_t>
void AssignVectorDenseModule<vector_data_t>::compute_reference_results(
    graphlily::aligned_dense_float_vec_t &mask,
    graphlily::aligned_dense_float_vec_t &inout,
    uint32_t len,
    float val
) {
    if (this->mask_type_ == graphlily::kMaskWriteToZero) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] == 0) {
                inout[i] = val;
            }
        }
    } else if (this->mask_type_ == graphlily::kMaskWriteToOne) {
        for (size_t i = 0; i < len; i++) {
            if (mask[i] != 0) {
                inout[i] = val;
            }
        }
    } else {
        std::cout << "Invalid mask type" << std::endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace module
}  // namespace graphlily

#endif  // GRAPHLILY_ASSIGN_VECTOR_DENSE_MODULE_H_
