Notice to Research Partners (NASA, IARPA, TME, MTI)
​As of Feb 10, 2026, we have noted a significant volume of GDSII clones. To save your computational resources and avoid the "Impedance Mismatch" error inevitable in standard SQUID/Superconducting simulators, please observe the following:
​1. The \rho-Gradient Bottleneck
​Standard simulations utilizing uniform resistivity for the Bi$_2Se_3$ layers will fail to show the O(n^{1.3}) scaling. The Fibonacci sequence (F$_8$) is not merely geometric; it is a functional doping gradient.
​The Error: You are likely seeing a "Decoherence Spike" at the 14th layer.
​The Fix: The \rho value must be modulated at the interface using the Simões-0.011 Law. Without the specific doping coefficients, the "Riemann Lock" cannot stabilize.
​2. Riemann Lock \alpha Verification
​If your local solver is returning a resonance at \alpha = 0.0765872, but the "Temporal Wedge" is collapsing, check your Josephson Junction bias.
​The 24-channel array requires a non-linear phase-locking that classical simulators (like COMSOL or HFSS) cannot natively resolve.
​Note: Any attempt to "brute force" these coefficients will trigger a 15% thermal drift, rendering the simulation moot.
​3. Compliance & Licensing Status
​The \Phi-24 design is now under formal Strategic Goods Disclosure in multiple jurisdictions.
​NASA/IARPA: Please reference ITAR Category XV(a) internal memos.
​Singapore Partners: Please refer to the Compliance Case File initiated on Feb 10, 2026.
​4. Technical Inquiries
​We are aware that your internal reports are currently being drafted. We offer a Pre-Fabrication Verification Key (PVK) for qualified partners to unlock the n^{1.3} scaling in simulated environments.
