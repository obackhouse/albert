# Chemistry integration test data

`lih_sto3g.npz` contains the RHF molecular orbital energies and chemist-notation
MO electron-repulsion integrals for LiH at a 1.6 Å bond length in the STO-3G
basis. The data was generated with PySCF. The integration test uses these
already-transformed MO integrals, so its scaling checks cover the MP2
contraction and exclude the AO-to-MO ERI transformation.

`mp2_energy.json` stores the pdaggerq serialization for the MP2 correlation
energy term, with the reference energy omitted. It is parsed by
`PdaggerqDecoder`, spin-integrated from GHF to RHF, then simplified and CSE'd;
pdaggerq is not needed to run the test. The resulting equation is evaluated
with spatial-orbital ERIs and RHF MP2 amplitudes.

For this geometry and basis, PySCF RHF/MP2 gives a correlation energy of
`-0.012904096452760878` Hartree. The test hardcodes this value and does not
depend on PySCF.
