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

COSMO-mic is an extension of COSMO-RS for studying micelles. It achieves this by
treating micellar systems as layered liquids, where each layer is a shell whose
shape matches that of the micelle. Each layer's composition, with respect to
each of the surfactant's atoms and ions, must be known, either from experimental
data or simulation techniques. This composition is translated to an atomic
probability distribution for each layer, which can be combined with the
COSMO-calculated surface charge densities for each atom in order to calculate a
layer-wise surface charge density profile. Finally, the match between this and
the surface charge density profile of a solute molecule indicates the chemical
potential of introducing the solute to that layer. This means that we can
determine the chemical potential of introducing another surfactant, A, to a
micelle that is composed of A. From this, the CMC can be calculated.

Alternatively, COSMO-RS can be used to determine CMCs solely by considering
homogeneous phase equilibria, without requiring knowledge of the internal
micellar structure. Andersson et al. developed two strategies for predicting
CMCs in this way:

For surfactants whose head group polarity is relatively low (most nonionic
surfactants), their strategy was to consider a binary system containing pure
phases of water, and of surfactant. COSMO-RS was employed to determine the molar
fraction of surfactant in the water phase at equilibrium, which was taken as the
CMC. This strategy considers a micelle as equivalent to a bulk surfactant phase;
in reality, a micelle's structure arises from the fact that it exists at an
interface. This approximation is more accurate when the interactions between
head and tail groups are balanced.

Their strategy for highly polar head groups (primarily ionic surfactants)
considers a ternary system of surfactant, oil and water, where the oil matches
the surfactant's tail chemistry. This constitutes two phases: pure oil, and a
surfactant/water mixture. The interfacial tension between the phases is computed
using COSMO-RS. The CMC is determined by iterating the aqueous concentration of
surfactant until the IFT is zero. Here, the premise is that the oil phase
represents the interior of the micelle, composed of tail groups, and the
phenomenon under consideration is the interaction of surfactant tail groups.
This is prescient in the case when the interactions between head and tail group
are particularly unfavourable due to their difference in polarity.

For practical applications, both strategies can be applied and the lower CMC
prediction should be used.

COSMOplex is an extension of COSMOmic that employs the fact that, in matching up
the surface charge profiles of micellar layers and a solute, COSMOmic calculates
a probability distribution of how molecules of that solute will arrange
themselves in the micelle. COSMOplex combines this distribution with a system
pressure term in order to predict the structure of the micelle, whose layer-wise
atomic distributions are computed and fed to COSMOmic, which generates a new
input for COSMOplex, and this process iterates until self-consistency is
achieved. In this way, an initial MD simulation does not need to be performed in
order to acquire the micellar structure.

There are four papers using COSMOmic or COSMOplex to determine CMCs for various
surfactants to look at, see Zotero. One obvious drawback using COSMOmic is that
it cannot accurately model the zeta potential, which is a problem for ionic
surfactants with counterions in solution.
