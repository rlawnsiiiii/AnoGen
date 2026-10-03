"""Torch-free mechanism testbed for the GenFSDiff sampler.

Exact linear-Gaussian denoisers stand in for the trained score so that the
*information structure* of a backbone (bidirectional vs causal) can be studied
in isolation, with exact Jacobians and no training noise. See
``docs/TESTBED.md``.
"""
