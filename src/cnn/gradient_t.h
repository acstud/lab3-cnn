#pragma once

// this is the grdient number in a tensor
struct gradient_t {
    float grad;
    float oldgrad;

    gradient_t() {
        grad = 0;
        oldgrad = 0;
    }
};
