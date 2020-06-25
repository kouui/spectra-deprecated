Attention
======================


Convention of wavelength-frequency conversion
---------------------------------------------------

for line profiles we assume :
    .. math::
        d\nu = -\frac{c}{\lambda^{2}}d\lambda \approx -\frac{c}{\lambda_{0}^{2}}d\lambda

therefore, we have
    .. math::
        \frac{d\nu}{\nu_{0}} \approx \frac{d\lambda}{\lambda_{0}}

and for wavelength/frequency mesh conversion, we have
    .. math::
        d\lambda = - \frac{c}{\nu_{0}^{2}} d\nu

for conversion between intensities in wavelength/frequency units, we have
    .. math::
        I_{\nu} d\nu = -I_{\lambda} d\lambda

        I_{\lambda} = I_{\nu} \frac{c}{\lambda_{0}^{2}}

for *Voigt function*, we have
    .. math::
        \int_{-\infty}^{+\infty} H(a,x) dx = \sqrt{\pi}

where
    .. math::
        x \equiv \frac{(\nu-\nu_{0})}{\Delta\nu_{D}} \approx \frac{(\lambda-\lambda_{0})}{\Delta\lambda_{D}}

and :math:`\Delta\nu_{D}`, :math:`\Delta\lambda_{D}` are *Doppler Width* in frequency, wavelength unit, respectly, since
    .. math::
        \frac{\Delta\lambda_{D}}{\lambda_{0}} = \frac{\Delta\nu_{D}}{\nu_{0}}
