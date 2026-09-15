# Physics-Informed Repulsion-Force Models for Simulating Traffic Yielding to Emergency Vehicles
### A Literature Review

**Scope.** This review covers the theoretical and applied literature relevant to modeling how surrounding traffic gives way to an emergency vehicle (EV) using force- and field-based ("physics-informed") approaches. It is organized in five strands: (1) foundational force-based microscopic models, (2) social force models extended to vehicles, (3) artificial potential field (APF) and driving-risk-field models, (4) emergency-vehicle-specific yielding and rescue-lane research, and (5) physics-informed machine learning hybrids. A synthesis section identifies gaps relevant to implementing EV-yielding behavior in a SUMO/Unreal Engine training simulator.

---

## 1. Foundational force-based microscopic models

**[1] Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots." *The International Journal of Robotics Research*, 5(1), 90–98.**
The origin of the artificial potential field paradigm in robotics: obstacles emit repulsive potentials and goals emit attractive potentials, and the robot descends the gradient of the summed field in real time. Nearly all vehicle-level repulsion models trace back to this formulation, including its known pathologies (local minima, oscillation near obstacles) which remain relevant when designing EV repulsion fields.

**[2] Helbing, D., & Molnár, P. (1995). "Social force model for pedestrian dynamics." *Physical Review E*, 51(5), 4282–4286.**
The seminal social force model (SFM): each agent is a Newtonian particle subject to a driving force toward a desired velocity (with relaxation time τ) plus exponentially decaying repulsive forces from other agents and boundaries. Crucially for EV modeling, it introduces anisotropy — stimuli in the agent's field of view exert stronger influence than those behind — which is the standard mechanism for making cars ahead of a siren react more strongly than cars behind it.

**[3] Helbing, D., Farkas, I., & Vicsek, T. (2000). "Simulating dynamical features of escape panic." *Nature*, 407, 487–490.**
Extends the SFM with physical contact forces and panic dynamics, demonstrating that force superposition can reproduce emergent collective phenomena (arching, clogging, herding) in high-urgency situations. Methodologically important as the template for modeling *urgency-modulated* behavior — the same mechanism by which siren proximity can scale repulsion gains in an EV-yielding model.

**[4] Treiber, M., Hennecke, A., & Helbing, D. (2000). "Congested traffic states in empirical observations and microscopic simulations." *Physical Review E*, 62, 1805–1824.**
Introduces the Intelligent Driver Model (IDM), the reference physics-based car-following model with interpretable parameters (desired speed, time headway, comfortable deceleration). IDM is the default longitudinal backbone that force-based lateral yielding models are typically layered on top of, and it is one of the physics priors used in later physics-informed deep learning work [17].

**[5] Kesting, A., Treiber, M., & Helbing, D. (2007). "General Lane-Changing Model MOBIL for Car-Following Models." *Transportation Research Record*, 1999(1), 86–94.**
MOBIL frames lane changes as an incentive/politeness trade-off computed from the accelerations of neighboring vehicles under a car-following model. It represents the discrete, rule-based alternative to continuous force fields for lateral behavior; hybrid EV-yielding implementations often use force fields to set the *target* and MOBIL-like safety criteria to gate the *maneuver*.

## 2. Social force models extended to vehicles

**[6] Huang, W., Fellendorf, M., & Schönauer, R. (2012). "Social Force Based Vehicle Model for Two-Dimensional Spaces."**
One of the first systematic transfers of the SFM to motor vehicles, adding what pedestrians lack: non-holonomic kinematics, steering-angle dynamics, and target forces split into location and direction components. Demonstrates that vehicle motion in 2D (unconstrained by lanes) can be produced by superimposed forces — directly relevant when NPC cars must leave their lane corridor to clear space.
https://transp-or.epfl.ch/heart/2012/latsis2012_submission_34.pdf

**[7] Pascucci, F., Rinke, N., Schiermeyer, C., Friedrich, B., & Berkhahn, V. (2015). "Modeling of shared space with multi-modal traffic using a multi-layer social force approach." *Transportation Research Procedia*, 10, 316–326.**
Applies the SFM to mixed car–bicycle–pedestrian shared spaces, where no lane discipline exists and all interaction is force-mediated. Shows the practical calibration issues (force parameter identifiability, oscillations) that arise when the SFM governs heavy, fast agents rather than pedestrians.

**[8] Rinke, N., Schiermeyer, C., Pascucci, F., Berkhahn, V., & Friedrich, B. (2017). "A multi-layer social force approach to model interactions in shared spaces using collision prediction." *Transportation Research Procedia*, 25, 1249–1267.**
Extends [7] with a collision-prediction layer: repulsive forces are triggered not by current proximity but by *anticipated* conflict points. Conceptually this is the precursor of corridor-based EV repulsion — repelling from a predicted trajectory rather than from a point position — which is what produces early, smooth yielding rather than last-moment evasion.

**[9] Yang, D., Özgüner, Ü., & Redmill, K. (2020). "A Social Force Based Pedestrian Motion Model Considering Multi-Pedestrian Interaction with a Vehicle." *ACM Transactions on Spatial Algorithms and Systems*, 6(2), Article 11.**
Models the vehicle as a dominant field source inside an SFM and calibrates the vehicle-influence force from real trajectory data using genetic algorithms. Useful as a methodological template for calibrating an EV's repulsion parameters (strength A, decay length B, anisotropy λ) against observed yielding trajectories rather than hand-tuning them.

## 3. Artificial potential fields and driving risk fields for road vehicles

**[10] Gerdes, J. C., & Rossetter, E. J. (2001). "A Unified Approach to Driver Assistance Systems Based on Artificial Potential Fields." *Journal of Dynamic Systems, Measurement, and Control*, 123(3), 431–438.**
Brings APF from robotics into vehicle dynamics and driver assistance: lane keeping and hazard avoidance are expressed as potential functions whose gradients act as forces on the vehicle model, with later work adding Lyapunov-based stability guarantees. Establishes the control-theoretic footing (boundedness, stability of the force-to-steering mapping) that a game-engine implementation should respect to avoid oscillating NPCs.

**[11] Wolf, M. T., & Burdick, J. W. (2008). "Artificial potential functions for highway driving with collision avoidance." *IEEE ICRA 2008*, 3731–3736.**
Designs structured highway potentials: anisotropic (elongated) vehicle repulsion shaped like a wake behind and ahead of each car, lane-center attraction, and road-edge repulsion. The elongated, velocity-dependent obstacle potential is essentially the shape needed for an EV field that clears the corridor ahead of the siren without disturbing traffic behind it.

**[12] Wang, J., Wu, J., & Li, Y. (2015). "The Driving Safety Field Based on Driver–Vehicle–Road Interactions." *IEEE Transactions on Intelligent Transportation Systems*, 16(4), 2203–2214. DOI: 10.1109/TITS.2015.2401837.**
The influential "driving safety field" theory: total driving risk is a unified field composed of a potential field (static objects), a kinetic field (moving objects, scaled by mass and velocity), and a behavior field (driver characteristics). An EV with lights and siren maps naturally onto this framework as a high-intensity kinetic field source whose "virtual mass" is inflated by its priority status.

**[13] Wang, J., Wu, J., Zheng, X., Ni, D., & Li, K. (2016). "Driving safety field theory modeling and its application in pre-collision warning system." *Transportation Research Part C*, 72, 306–324.**
Generalizes [12] into a modified, tailorable field model and validates it in a multi-vehicle pre-collision warning application with field experiments. Demonstrates the pipeline from field formulation → risk index → actionable vehicle behavior, which is the same pipeline an NPC yielding controller implements (field → force → lateral/longitudinal command).

**[14] Rasekhipour, Y., Khajepour, A., Chen, S., & Litkouhi, B. (2017). "A Potential Field-Based Model Predictive Path-Planning Controller for Autonomous Road Vehicles." *IEEE Transactions on Intelligent Transportation Systems*, 18(5), 1255–1267.**
Embeds potential functions inside an MPC cost rather than integrating raw forces, separating obstacle classes (crossable vs. non-crossable) with different potential shapes. Relevant as the "stable" alternative when direct force integration produces jitter: the field defines costs, and an optimizer produces dynamically feasible yielding trajectories.

**[15] "Probabilistic field approach for motorway driving risk assessment." *Transportation Research Part C* (2020).**
Formulates the risk field objectively as collision probability × expected crash energy over probabilistically predicted motion, addressing the criticism that classical safety-field parameters are not empirically grounded. Points toward making EV repulsion *probabilistic*: cars repel from the distribution of the EV's likely future positions, not a single deterministic path.
https://www.sciencedirect.com/science/article/pii/S0968090X20306318

**[16] Ling, R., Li, M., Liu, B., & Li, Z. (2024). "A Vehicle Size-Based Dynamic Model of Artificial Driving Risk Potential Fields and Vehicle Interaction Analysis for Highway Driving." arXiv:2412.19190.**
Introduces vehicle dimensions into the risk potential field: larger and faster vehicles generate stronger, wider fields that extend into adjacent lanes and measurably alter neighbors' motion. Directly transferable to EVs — a fire truck should project a physically larger and stronger field than an ambulance — and provides contour analyses useful for parameterizing field extent by vehicle class.

## 4. Emergency-vehicle-specific yielding and rescue-lane modeling

**[17] Bieker-Walz, L. (2018). "Analysis of the traffic behavior of emergency vehicles in a microscopic traffic simulation." *SUMO 2018 Conference Proceedings (EPiC Series in Engineering, vol. 2)*, 1–13.**
The core SUMO-native reference: implements EV special rights via a "blue light device," makes surrounding traffic form a rescue lane, and evaluates simulated EV driving against real-world EV data (noting simulated vehicles clear intersections somewhat faster than reality). This is the baseline behavior that a physics-informed repulsion layer would replace or refine, and the paper documents its limitations.
https://easychair.org/publications/paper/dMh9

**[18] Cortés, C. E., et al. (2023). "Trajectory Simulation of Emergency Vehicles and Interactions with Surrounding Traffic." *Journal of Advanced Transportation*. DOI: 10.1155/2023/5995950. (See also Cortés & Stefoni, SSRN 4125888.)**
Develops explicit non-EV reaction models — lane changing in response to an EV, sidewalk mounting under congestion, and intersection-approach behavior — implemented via an API on top of commercial microsimulation and calibrated with fire-truck GPS traces and onboard video from Santiago. One of the few works with *empirically calibrated yielding behavior*, and a strong argument that default simulator behavior substantially mispredicts EV travel paths and times.

**[19] Hannoun, G. J., Murray-Tuite, P., Heaslip, K., & Chantem, T. (2019). "Facilitating Emergency Response Vehicles' Movement Through a Road Segment in a Connected Vehicle Environment." *IEEE Transactions on Intelligent Transportation Systems*, 20(9), 3546–3557. DOI: 10.1109/TITS.2018.2877758.**
The optimization-based counterpoint to force models: an integer linear program assigns pull-over positions that maximize distance between non-EVs and the EV's path in a connected-vehicle setting. Useful as an *optimal benchmark* — a repulsion-force model can be evaluated by how closely its emergent clearance approaches the ILP-optimal clearance.

**[20] Wu, J., Kulcsár, B., Ahn, S., & Qu, X. (2020). "Emergency vehicle lane pre-clearing: From microscopic cooperation to routing decision making." *Transportation Research Part B*, 141, 223–239.**
Models cooperative *pre-clearing*, where vehicles vacate the EV lane early enough that the EV never decelerates, and links the microscopic cooperation model to routing decisions. Highlights the timing dimension mostly absent from static field models: when repulsion begins (siren detection distance) matters as much as its spatial shape.

**[21] Kuzmic, J., & Rudolph, G. (2020). "Unity 3D Simulator of Autonomous Motorway Traffic Applied to Emergency Corridor Building." *(IoTBDS/SIMULTECH proceedings)*.**
Builds a game-engine (Unity 3D) motorway simulator specifically to study automated rescue-corridor (Rettungsgasse) formation, encoding lane-width-aware lateral offsets and V2V/I2I communication levels. The closest methodological relative to a UE5-based training simulator, and a demonstration that corridor formation can be studied credibly inside a game engine rather than a classical microsimulator.
https://www.researchgate.net/publication/341470027

**[22] Su, H., Shi, K., Chow, J. Y. J., & Jin, L. (2021). "Dynamic queue-jump lane for emergency vehicles under partially connected settings: A multi-agent deep reinforcement learning approach." arXiv:2003.01025.**
Uses multi-agent deep RL so that connected vehicles learn to open a dynamic queue-jump lane for an approaching EV under partial connectivity. Represents the learning-based alternative to hand-designed repulsion; its reward shaping (EV progress vs. traffic disruption) is a useful objective function even for tuning analytic force parameters.

**[23] Su, H., et al. (2022). "EMVLight: A Multi-agent Reinforcement Learning Framework for an Emergency Vehicle Decentralized Routing and Traffic Signal Control System." arXiv:2206.13441.**
Couples EV routing with decentralized traffic-signal control in an MARL framework, reducing EV travel time while limiting impact on background traffic. Relevant for the network-level layer above vehicle-level repulsion: signals and routing shape where yielding interactions occur.

## 5. Physics-informed learning hybrids

**[24] Mo, Z., Shi, R., & Di, X. (2021). "A physics-informed deep learning paradigm for car-following models." *Transportation Research Part C*, 130, 103240. arXiv:2012.13376.**
Defines the PIDL-CF paradigm: neural car-following models regularized/encoded with IDM and OVM physics, improving data efficiency and preventing physically implausible predictions across acceleration, deceleration, cruising, and emergency-braking regimes. This is the canonical recipe for the "physics-informed" ambition — an analytic EV repulsion model can serve as the physics prior, with a neural residual learned from real yielding trajectories.

**[25] Ma, L., Qu, S., Song, L., Zhang, Z., & Ren, J. (2023). "A Physics-Informed Generative Car-Following Model for Connected Autonomous Vehicles (PICGAN)." *Entropy*, 25(7), 1050. DOI: 10.3390/e25071050.**
Embeds physics-based structure in a conditional GAN for multi-step trajectory generation in mixed traffic (NGSIM I-80), removing the explicit weighting between physics and data terms. Relevant when the goal is *generating* diverse, realistic yielding trajectories for NPC populations rather than predicting a single deterministic response.

## 6. Simulation infrastructure (supporting references)

**[26] Lopez, P. A., et al. (2018). "Microscopic Traffic Simulation using SUMO." *IEEE ITSC 2018*.** — The reference paper for SUMO, including TraCI online interaction; the platform in which lane-based EV yielding (blue-light device, sublane model) is implemented.

**[27] Erdmann, J. (2015). "SUMO's Lane-Changing Model." In *Modeling Mobility with Open Data*, Springer.** — Documents the lane-change layer (and the basis of the sublane extension) that any force-to-SUMO mapping must interface with via `setLateralLanePosition`, `setLaneChangeMode`, and related TraCI calls.

---

## 7. Synthesis and research gaps

**Convergent findings.** Across strands 1–3, a consistent architecture emerges: superimpose (i) a goal/driving force, (ii) anisotropic, velocity-dependent repulsion from other agents, and (iii) boundary forces — then either integrate forces directly (SFM tradition [2,6–9]) or use the field as a cost inside planning/control (APF-MPC tradition [10,11,14]). The driving-safety-field line [12,13,15,16] adds empirically motivated field *shapes* (mass-, speed-, and size-scaled), which is exactly the knob needed to make an emergency vehicle a privileged, high-intensity field source.

**EV-specific evidence.** The EV literature shows that (a) default microsimulator yielding is unrealistic and must be extended [17,18]; (b) empirical calibration from EV-mounted sensors is feasible and materially changes results [18]; (c) optimal clearance benchmarks exist [19,20]; and (d) game engines are viable platforms for rescue-corridor studies [21].

**Identified gaps relevant to a SUMO+UE5 training simulator:**
1. **No published anisotropic repulsion-field model repels from the EV's *predicted corridor*** (route-projected polyline) rather than its position; the collision-anticipation layer of [8] and probabilistic fields of [15] point in this direction but have not been combined with EV priority semantics.
2. **Siren/perception onset is under-modeled.** Field models are geometric; [20] shows timing dominates outcomes, yet detection distance, acoustic occlusion, and driver reaction latency rarely parameterize the field's activation.
3. **Calibration data for yielding is scarce.** Only [18] calibrates non-EV reactions against real EV trips; no public dataset supports fitting (A, B, λ)-type repulsion parameters for European rescue-lane behavior, suggesting a PIDL approach [24] with an analytic repulsion prior as a data-efficient path.
4. **Force-to-lane-based mapping is ad hoc.** The literature either abandons lanes entirely (SFM in shared space) or stays lane-discrete (MOBIL, SUMO); a principled projection of continuous repulsion forces onto SUMO's sublane/lateral dynamics — with stability guarantees in the spirit of [10] — is an open implementation question.

---

*Compiled 14 July 2026. DOIs/URLs given where verified; classic references [1–5, 10, 11, 26, 27] are standard works cited in the retrieved literature.*
