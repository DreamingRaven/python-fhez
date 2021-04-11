.. include:: substitutions

.. |hadmard-fig| image:: img/hadmard-product.svg
  :width: 400
  :alt: Hadmard product of two 2D matrices

.. |cnn-fig| image:: img/cnn.svg
  :width: 700
  :alt: Full CNN computational graph

.. |cnn-ann-loss-fig| image:: img/cnn-ann-loss.svg
  :width: 700
  :alt: Full end-to-end cnn-ann-loss computational graph

.. |kernel-masquerade-fig| image:: img/kernel-masquerade.svg
  :width: 700
  :alt: Kernel masquerading as a mask.

.. _section_cnn:

Convolutional Neural Network (CNN)
##################################

|cnn-fig|

Convolutional neural networks are quite complicated cases with FHE.
Since the encrypted cyphertext is the most atomic form of the input data that we can access and we need to be able to multiply subsets of the data by different amounts we use a sparse n-dimensional array with the weights embedded in different positions and the rest zeroed out (see |section_masquerade| and |section_hadmard_product|). This way we can still convolve out filters/ kernels but instead of manipulating :math:`x` (normally by selecting a slice of :math:`x` that you were interested in) we manipulate the filter instead generating windows where the filter should be placed, and caching these windows for later use in back-propogation so we know exactly what input :math:`x` multiplied which weights once :math:`x` is finally decrypted. Each window becoming a different branch :math:`<t>` and the total number of windows our total branches :math:`T_x`.


.. _section_cnn_equations:

CNN Equations
+++++++++++++

Standard neuron equation:

.. math::
  :label: cnn

  a = g(\sum_{i=0}^{n-1}(w_ix_i)+b)

|section_masquerade|, weighted, and biased cross correlation.

.. math::
  :label: cnn-commuted

  a^{(i)<t>} = g(\sum_{t=0}^{T_x-1}(k^{<t>}x^{(i)})+b/N)

CNN Derivatives
---------------

The derivative of a CNN (:math:`f`) with respect to the bias :math:`b`:

.. math::
  :label: cnn-dfdb

  \frac{df(x)}{db} = T_x \frac{dg}{dx}

The derviative of a CNN (:math:`f`) with respect to the weights multi-dimensional array :math:`w` is the sum of all portions of :math:`x^{(i)}` unmasked during product calculation:

.. math::
  :label: cnn-dfdw

  \frac{df(x)}{dw} = \sum_{t=0}^{T_x}(x^{(i)<t>})\frac{dg}{dx}

The derivative of a CNN (:math:`f`) with respect to the input :math:`x`:

.. math::
  :label: cnn-dfdx

  \frac{df(x)}{dx} = \sum_{t=0}^{T_x}(k^{(i)<t>})\frac{dg}{dx}

.. note::

  .. include:: variables


.. _section_masquerade:

Kernel-Masquerade
+++++++++++++++++

The |section_masquerade| is the combining of a convolutional kernel and a *mask* to simultaneously calculate the product of any given kernel/ filter over a dataset. This is to *act* as though we were capable of normally slicing some input data which is impossible when this data is embedded in the FHE cyphertext. We deploy kernel weights embedded in a sparse/ zeroed multi-dimensional array, thus ignoring undesired values in the input cyphertext when finding the |section_hadmard_product| of the two, and minimising computational depth of cyphertexts in processing.

|kernel-masquerade-fig|

.. _section_hadmard_product:

Hadmard Product
+++++++++++++++

The Hadmard product is simply two equally shaped n-dimensional arrays operated on element wise to produce a third product n-dimensional array of the same size/ shape:

|hadmard-fig|


.. _section_commuted_sum:

Commuted-Sum
++++++++++++

.. warning::

  If you are intending to write your own or extend an additional neural network please pay special attention to the |section_commuted_sum|, it will change the behavior of the networks drastically and in some unexpected ways if it is not accounted for. CNNs and other single input multiple branch networks are the source of commuted-sums.

In our neural networks operating on fully homomorphically encrypted cyphertexts, the cyphertexts encode into themselves multiple values, I.E for us they include **all** the values in a single training example :math:`(i)`, to give a more concrete example, for a CNN one cyphertext is the **whole** of a single image, thats all pixel values of the image. This means computations happen quicker and take up less space as there is less overhead since we do not need different parameters between each pixel value, instead we can encode and encrypted them all using the same parameters, reducing duplication, and allowing us to operate on all of them simultaneously. A consequence to this approach however is that the cyphertext is the most atomic form of the data we have, it cannot be any smaller, it will always be a polynomial with :math:`N` number of values encoded in it. In practice this means we cannot sum a cyphertext, as it would mean reducing all these values into one single value I.E the sum of the cyphertext, or to put it a different way, we cannot fold the cyphertext in on itself to get this one single number, as this is *homomorphic* encryption I.E structure preserving self similar encryption, the form of what comes in is the form of what comes out.

In the case of CNNs however as you can see in |section_cnn_equations| there is a sum. As we have stated it is impossible to sum the cyphertext in on itself, thus since almost all our networks *have* to be/ use abelian operations we can *commute* this sum until after we have decrypted as long as we pay special attention to the fact that we have big matrices that should be treated as if they were singular values. The most egregious possible violation is broadcasting of a single addition, as this acts differently on a single value than it does on a matrix of values, thus the need to always divide a bias :math:`b` by the total number of values being biased :math:`N`. Assuming our calculus is correct and we succeffully implement a commuted sum until after decryption, this should only ever be a problem in one dimension, all subsequent sums would happen between cyphertexts (since it should have been summed already) meaning there is only ever one commuted sum, we never have to worry about a stack of them.

|cnn-ann-loss-fig|

CNN API
-------

.. note::

  |api-build-note|

.. autoclass:: fhe.nn.layer.cnn.Layer_CNN
  :members:
