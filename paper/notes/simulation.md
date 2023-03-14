Predicting CMC with simulations
===============================

Discuss two useful approaches:

- COSMO-RS-based methods.
- DPD-based methods.

COSMO-RS-based approaches
-------------------------

COSMO (Conductor-like screening model) treats a solute as a cavity in a
dielectric continuum solvent; in other words, as a volume that is capable of
screening charge. The approach is based on calculating the polarisation charges
of the solvent-accessible surface of the molecule by splitting this area into
individual segments and applying a charge-scaling rule. This approach allows us
to calculate the free energy of solvation relatively cheaply by considering the
total energy of this vdW-like surface.

An advantage of this type of inspection over molecular dynamics is that it is
hard to account for polarisation effects and directed forces from
dipole/quadrupole moments, such as lone pairs (from COSMO-RS paper) using
traditional force fields. For polar molecules, most of the solvation effects
arise from charge screening, so this is a serious limitation.

COSMO-RS is an extension of COSMO that accounts for "real solvents", i.e. ones
that do not act as conductors, screening molecules in an ideal way. In COSMO,
each segment of the SAS is assumed to be matched exactly with an opposite
surface charge density by the solvent. COSMO-RS considers non-ideal pairings of
surface charge density, as well, in order to account for entropy considerations
as well as the fact that many molecules' surface charge densities do not
"match-up": they cannot pair ideally. COSMO-RS applies statistical mechanics to
compute the probability distribution of how surface charges between molecules
align, from which the chemical potential of solvation can be determined.
