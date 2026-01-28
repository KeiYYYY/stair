# Mathematical Modelling of Staircase Wear: A Tribological and Pedestrian Dynamics Approach to Archaeological Reconstruction

## Executive Summary

The erosion of staircase surfaces in heritage structures constitutes a cumulative, albeit complex, record of human activity, material interaction, and environmental exposure. This report presents a comprehensive mathematical framework designed to decode the historical data embedded in worn stair treads. By integrating principles from tribology (specifically Archard’s law of wear and hysteresis models), biomechanics (gait analysis, ground reaction forces, and shear stress distribution), and pedestrian dynamics (social force models, lane formation, and lateral distribution variance), we establish a robust, multi-scalar model for inferring usage patterns, age, and structural history from non-destructive measurements.

The proposed model addresses the "inverse problem" of wear: determining the historical input vectors (traffic volume $N(t)$, directionality vector $\vec{v}$, group dynamics $\rho$) based on the observed output state $S(t_{final})$ (wear depth $h(x,y)$ and topographical profile). We demonstrate that while mechanical wear is primarily driven by the integral of contact pressure and sliding distance over time, the critical distinction between "high traffic/short duration" and "low traffic/long duration" scenarios can be resolved by analyzing the differential coupling between mechanical abrasion and background chemical weathering. Furthermore, the cross-sectional geometry of wear patterns—specifically the kurtosis, skewness, and modality of the wear depth profile—serves as a reliable indicator of simultaneous usage (single-file vs. side-by-side) and bi-directional flow dynamics.

We provide a detailed field guide for archaeologists to acquire necessary data using low-cost, non-destructive techniques such as photogrammetry, Schmidt hammer rebound testing, and portable X-ray fluorescence (pXRF). The report concludes with specific algorithmic approaches for detecting renovations through discontinuity analysis and identifying material provenance via statistical clustering of trace elements.

------

## 1. Introduction: The Staircase as a Palimpsest

Stairs are among the most enduring architectural elements, often surviving the collapse of roofs and the crumbling of walls. They are functional necessities that strictly dictate human movement, forcing individuals to interact with the built environment in a highly constrained and repetitive manner. This mechanical repetition turns the staircase into a palimpsest—a surface that records the history of its use through the gradual, stochastic subtraction of material.

However, deciphering this record is non-trivial. The observed wear on a step is the integral of centuries of stochastic events: millions of footfalls, varying footwear materials (from leather sandals to hobnailed boots), environmental cycles (freeze-thaw, acid rain), and maintenance interventions. The problem posed—to determine the age, traffic patterns, and construction history of a stairwell from its wear profile—requires a multidisciplinary approach that bridges the gap between physical mechanics and behavioral science.

### 1.1 The Nature of the Inverse Problem

The core challenge is an inverse problem in mathematical modeling. We are presented with the final state $S(t_{final})$—the worn geometry—and must reconstruct the history function $H(t)$ that produced it. This reconstruction is complicated by several distinct factors:

1. **Equifinality:** Different combinations of traffic intensity ($I$) and duration ($T$) can theoretically produce similar total wear volumes ($V$). A stair used by 1,000 people a day for 10 years may exhibit a similar *maximum depth* of wear to one used by 10 people a day for 1,000 years. Disentangling these requires analyzing secondary signatures such as edge weathering and micro-pitting.
2. **Material Heterogeneity:** Natural stones are not isotropic. Variations in mineral hardness (e.g., the differential erosion of quartz vs. feldspar in granite) lead to non-uniform wear rates.
3. **Variable Boundary Conditions:** The use of a building evolves. A temple may become a tourist site; a fortress may become a residence. These transitions alter traffic patterns, shifting the "desire lines" and modifying the wear distribution.

### 1.2 Objectives and Scope of the Model

This report develops a predictive model that allows archaeologists to:

- **Quantify Traffic:** Establish a mathematical relationship between wear depth/volume and cumulative pedestrian traffic (footsteps).
- **Discern Directionality:** Use topographical asymmetry (skewness along the $y$-axis) to distinguish between ascent-dominant and descent-dominant flow.
- **Estimate Social Density:** Utilize lateral wear profile variance (kurtosis and bimodality along the $x$-axis) to estimate group sizes and lane formation behavior.
- **Detect Discontinuities:** Identify temporal anomalies indicative of renovation, repair, or material replacement.

The scope encompasses mathematical derivations of wear laws, simulation strategies for pedestrian dynamics using agent-based logic, and practical field protocols. While the principles apply to wood, the primary focus is on stone due to its prevalence in archaeological contexts.

------

## 2. Theoretical Framework: The Physics of Wear (Tribology)

To model the erosion of stairs, we must first establish the physical laws governing the removal of material. The primary mechanism is **abrasive wear**, where hard asperities (sand, grit, shoe soles) slide across a softer surface (the stair tread), removing material via plastic deformation, brittle fracture, and grain plucking.

### 2.1 Archard’s Law of Wear: The Fundamental Equation

The foundational equation for this analysis is Archard’s Law, which posits that the volume of material removed ($V$) is proportional to the normal load ($W$) and the sliding distance ($L$), and inversely proportional to the hardness of the material ($H$).

$$V = K \frac{W \cdot L}{H}$$

Where:

- $V$ is the wear volume ($m^3$).
- $K$ is the dimensionless wear coefficient, representing the probability of debris formation per asperity contact.
- $W$ is the normal load (Newtons), derived from the pedestrian's body weight and dynamic Ground Reaction Forces (GRF).
- $L$ is the sliding distance ($m$), representing the aggregate microslip that occurs during the gait cycle.
- $H$ is the Vickers hardness of the wearing surface ($Pa$).

#### 2.1.1 The Spatiotemporal Differential Form

For a stair tread, wear is a function of position $\mathbf{x} = (x, y)$ and time $t$. We define the wear depth rate $\dot{h}(\mathbf{x}, t)$ as:

$$\frac{\partial h(\mathbf{x}, t)}{\partial t} = k_{spec} \cdot \iint P(\mathbf{x}, t) \cdot \mathbf{v}_{slip}(\mathbf{x}, t) \, d\mathbf{x} \, dt$$

Simplifying for a discrete event model where wear accumulates per step:

$$\Delta h(x,y) = k_{spec} \cdot p_{contact}(x,y) \cdot s_{slip}(x,y) \cdot N_{flux}$$

Where:

- $k_{spec} = \frac{K}{H}$ is the **Specific Wear Rate** ($m^2/N$), a fundamental material property.
- $p_{contact}(x,y)$ is the **contact pressure distribution** ($Pa$) exerted by a footstep at position $(x,y)$.
- $s_{slip}(x,y)$ is the **local sliding distance** ($m$) per step (microslip magnitude).
- $N_{flux}$ is the **traffic flux** (steps per unit time).

This formulation is critical because it moves beyond simple volume calculation to **topographical prediction**. It implies that wear depth is maximized not just where people step often ($N_{flux}$), but where the product of pressure and sliding (shear work) is highest.

### 2.2 Material Coefficients: The $k_{spec}$ Database

The variable $k_{spec}$ allows us to normalize wear across different materials. An archaeologist finding 5mm of wear on granite faces a very different historical reality than one finding 5mm of wear on sandstone.

**Table 1: Comparative Specific Wear Rates and Hardness for Stair Materials**



| **Material Class** | **Mohs Hardness** | **Vickers Hardness (H)** | **Specific Wear Rate (kspec)** | **Dominant Wear Mechanism**                                 | **Archaeological Implication**                               |
| ------------------ | ----------------- | ------------------------ | ------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| **Granite**        | 6-7               | ~6-9 GPa                 | Low ($10^{-6} mm^3/Nm$)        | Brittle fracture of Quartz/Feldspar; "Differential Erosion" | Extremely high traffic or immense age required for visible concavity. |
| **Marble**         | 3-4               | ~1.5-3 GPa               | Medium ($10^{-5} mm^3/Nm$)     | Plastic deformation, dissolution, scratching                | Moderate wear implies standard usage; highly susceptible to acid rain. |
| **Sandstone**      | 6-7 (grains)      | Variable (cement)        | High (Variable)                | Grain detachment (plucking)                                 | Rapid wear; "rounding" of edges happens quickly. Good for detecting short-term high traffic. |
| **Limestone**      | 3-4               | ~1-3 GPa                 | Medium-High                    | Abrasion and chemical dissolution                           | Polishes easily; wear tracks can become slippery, altering gait (feedback loop). |
| **Wood (Oak)**     | N/A               | Low                      | High (Anisotropic)             | Fibrous compression and tearing                             | Wear is highly dependent on grain direction. Susceptible to rot, requiring frequent replacement. |

**Insight - Differential Erosion:** In polymineralic rocks like granite, the softer feldspar matrix wears faster than the harder quartz crystals. This creates a rough, raised relief at the microscopic level. If a stair tread is smooth (polished), it suggests a wear regime dominated by fine abrasives (dust/sand) over a very long period. If it is rough, it may indicate "plucking" caused by high-impact footwear or recent, intense damage.

### 2.3 Coupled Erosion: Mechanical vs. Chemical

To address the prompt's question about distinguishing "high traffic/short time" from "low traffic/long time," we must expand the model to include non-anthropogenic factors.

Let total erosion $E_{total}(t)$ be the sum of mechanical wear $W_{mech}$ and chemical/environmental weathering $W_{env}$:

$$E_{total}(x,y,t) = \underbrace{k_{spec} \int_0^t N(\tau) \sigma_{shear}(x,y) d\tau}_{\text{Mechanical (Traffic)}} + \underbrace{\gamma_{env} \cdot t}_{\text{Chemical (Time)}}$$

Where $\gamma_{env}$ is the rate of environmental degradation (e.g., dissolution of limestone by acidic rain, or UV degradation of wood).

**The Discriminant Method:**

- **The Wear Track ($x_{center}$):** Dominated by mechanical wear. $E \approx W_{mech} + W_{env}$.
- **The Unworn Edge ($x_{edge}$):** Dominated by environmental weathering. $E \approx W_{env}$.

By measuring the material loss or surface roughness at the **untrodden edges** of the stair (near the walls or risers), an archaeologist can estimate $\gamma_{env} \cdot t$. This establishes a baseline "clock" for the structure's age, independent of traffic. Subtracting this baseline from the center wear gives the purely mechanical component, allowing for a more accurate estimate of $N$ (traffic volume).

------

## 3. Biomechanics of Stair Negotiation (The Meso Scale)

To populate the pressure distribution term $p(x,y)$ in our model, we must understand the biomechanics of stair ambulation. Moving vertically on stairs generates forces significantly different from level walking.

### 3.1 Ground Reaction Forces (GRF) and Shear

The mechanical energy available to remove material comes from the Ground Reaction Force vector $\mathbf{F}_{GRF}$.

1. **Vertical Force ($F_z$):**
   - **Ascent:** Characterized by a double-peak profile (loading response and push-off). Peak forces typically reach **1.1 to 1.3 x Body Weight (BW)**.
   - **Descent:** Characterized by a sharp, high-magnitude impact transient at heel strike. Peak forces can reach **1.5 to 2.0 x BW** due to the need to decelerate the body's center of mass.
2. **Shear Forces ($F_{shear}$):**
   - **Ascent (Propulsion):** The dominant shear force is posterior-directed as the foot plantarflexes ("toe-off") to propel the body upward. This concentrates high shear stress at the **center and rear** of the tread.
   - **Descent (Braking):** The dominant shear force is anterior-directed during weight acceptance to prevent the body from falling forward. This occurs primarily at the **nosing (leading edge)** of the tread.

**Insight - Topographical Asymmetry:** Because descent generates higher peak vertical forces and relies heavily on braking shear at the step edge, **descent contributes disproportionately to nosing wear**. A stair with a significantly rounded or beveled leading edge indicates heavy descent traffic. Conversely, a stair with a deep "cup" in the center but a relatively crisp leading edge suggests ascent dominance or bi-directional traffic where descent was cautious (slow).

### 3.2 Foot Placement Probability Density Functions

Pedestrians do not step in the same location every time. Foot placement is stochastic, described by a Probability Density Function (PDF) $\Phi(x,y)$.

#### 3.2.1 Longitudinal Distribution ($y$-axis)

The "Going" (tread depth) determines foot placement.

- **Descent:** To maximize stability, pedestrians place the ball of the foot near the edge. There is often "overhang" (toes extending beyond the nosing), reducing the contact area $A$ and increasing local pressure $P = F/A$.
- **Ascent:** Foot placement is deeper on the tread to ensure full support for the push-off phase.

**Mathematical Signature:**

Let $y=0$ be the riser and $y=G$ be the nosing.

- $\Phi_{ascent}(y)$ is skewed towards $y \approx G/2$ (Center).
- $\Phi_{descent}(y)$ is highly skewed towards $y \approx G$ (Edge).

#### 3.2.2 Lateral Distribution ($x$-axis)

Lateral placement is governed by the "Effective Width" and social dynamics.

- **Single File:** Unimodal Gaussian distribution centered at $\mu = W/2$.
- **Bi-Directional / Group:** Bimodal Gaussian mixture distribution.

------

## 4. Macroscopic Pedestrian Dynamics: The Traffic Model

Archaeologists are interested in *how* the stairs were used. Were people moving singly? In groups? In opposing lanes? We apply models from pedestrian dynamics to simulate these behaviors.

### 4.1 The "Effective Width" Concept

Pedestrians maintain a boundary layer clearance from walls and obstacles to avoid collision and accommodate body sway. This reduces the usable width of the stair.

$$W_{eff} = W_{nominal} - 2\delta$$

Where $\delta$ is the shy distance, typically **15-30 cm** for walls and **10-15 cm** for handrails.

- If $W_{eff} < 0.75m$: Traffic is strictly single-file. Wear will be a single, deep, central groove.
- If $W_{eff} > 1.2m$: Lane formation becomes possible.
- If $W_{eff} > 1.5m$: Two distinct lanes can form comfortably.

**Insight - The "W" Profile:** On wide stairs ($>1.5m$), dense bi-directional traffic spontaneously self-organizes into lanes to minimize conflict. This results in a bimodal wear pattern (a "W" shape in cross-section). If an archaeologist observes a "W" profile on a wide staircase, it confirms **high-volume, simultaneous bi-directional use**. If a wide staircase has only a single central wear trough, it implies **low-volume usage** where users preferred the geometric center (maximizing clearance) and rarely passed one another.

### 4.2 Traffic Flow and Density

The relationship between flow ($J$), density ($\rho$), and velocity ($v$) is described by the Fundamental Diagram:

$$J = \rho \cdot v(\rho)$$

On stairs, velocity decreases as density increases.

- **Free Flow (Low Density):** Users choose the optimal path (usually the center or the "inside" of a curved stair). $\sigma_{lateral}$ (variance of foot placement) is low.
- **Congested Flow (High Density):** Users are forced to use the entire effective width. $\sigma_{lateral}$ increases significantly.

**Mathematical Signature of Flux:**

- **High Kurtosis ($K > 3$):** Indicates "channeled" movement, typical of low density (freedom to choose the best line) or very narrow stairs.
- **Low Kurtosis ($K < 3$):** Indicates "dispersed" movement, typical of high-density crowd flow where people are forced to step closer to the walls.

------

## 5. The Comprehensive Mathematical Model

We synthesize the physical and behavioral components into a discrete grid-based simulation model suitable for archaeological reconstruction.

### 5.1 The Grid Accumulation Algorithm

Let the stair tread be discretized into a grid matrix $H$ of size $M \times N$, where each cell $h_{ij}$ represents the wear depth at coordinates $(x_i, y_j)$.

The accumulated wear depth after time $T$ is:

$$H_{total}(T) = \sum_{t=0}^{T} \sum_{k=1}^{N_{ped}(t)}  + \Gamma_{env}(T)$$

Where:

1. **Material Susceptibility ($\alpha_{mat}$):** Derived from $k_{spec}$ (Section 2.2).
2. **Biomechanical Intensity ($\beta_{k}$):** A scalar representing the "aggressiveness" of pedestrian $k$ (mass, shoe hardness, gait speed).
3. **Step Kernel ($\mathbf{S}_{k}$):** A matrix representing the pressure distribution of a single footstep, spatially translated to the pedestrian's chosen foot placement $(x_k, y_k)$.
   - $\mathbf{S}_{k}$ is modeled as a 2D Gaussian or a dual-ellipsoid (heel/toe) distribution.
4. **Environmental Decay ($\Gamma_{env}$):** Background weathering function (Section 2.3).

### 5.2 Gaussian Mixture Model (GMM) for Lateral Position

To determine the centroids $(x_k)$ for the Step Kernel, we model the lateral probability density function $P(x)$ as a Gaussian Mixture:

$$P(x | \theta) = \sum_{m=1}^{M} w_m \mathcal{N}(x | \mu_m, \sigma_m^2)$$

- **Model Inputs:** Stair width ($W$), Handrail presence, Traffic mode (One-way vs Two-way).
- **Model Outputs:** The parameters $\mu_m$ (lane centers) and $\sigma_m$ (lane widths).

**Inverse Solution Algorithm:**

Given the observed lateral wear profile $W_{obs}(x)$ (extracted from photogrammetry):

1. Fit a GMM to $W_{obs}(x)$ using Expectation-Maximization (EM).
2. **If $M=1$ (Unimodal):** Traffic was single-file or unidirectional.
3. **If $M=2$ (Bimodal):** Traffic was bi-directional or paired.
4. **Calculate Separation:** $\Delta \mu = |\mu_1 - \mu_2|$. If $\Delta \mu \approx 0.6-0.8m$, it confirms distinct lanes.

------

## 6. Addressing the "Hard Questions": Diagnostic Algorithms

The prompt requests guidance on specific challenging scenarios. We provide algorithmic approaches for each.

### 6.1 Guidance: Is the Wear Consistent? (Consistency Check)

**Algorithm:**

1. Normalize wear depth $d_{step}$ for every step in the flight.
2. Plot $d_{step}$ vs. Step Number ($n$).
3. **Theoretical Expectation:** Wear should follow a parabolic trend.
   - **Start/End Effects:** Wear is typically lower at the very top and bottom steps because people decelerate and place feet more carefully (lower shear).
   - **Mid-Flight:** Wear is highest in the middle steps where gait rhythm is established and speed is constant.
4. **Anomaly Detection:** Any step deviating by $> 2\sigma$ from this parabola is "inconsistent."
   - **Explanation:** A replaced step (lower wear), a localized weak stone (higher wear), or a landing that disrupts rhythm.

### 6.2 Guidance: Determining Age and Reliability

**Algorithm:**

1. **Primary Estimator (Wear Volume):**

   $$T_{est} = \frac{V_{measured}}{N_{est} \cdot V_{step}}$$

   - $V_{measured}$: Total volume of material lost (from photogrammetry).
   - $N_{est}$: Estimated daily traffic (from historical context).
   - $V_{step}$: Volume loss per step (from $k_{spec}$ tables).
   - *Reliability:* Low. Highly sensitive to errors in $N_{est}$.

2. **Secondary Estimator (Weathering Rind):**

   Using the **Schmidt Hammer** on the **riser** (vertical face, unworn).

   $$R_{weathered} = R_{fresh} - f(T)$$

   - Compare the rebound value ($R$) of the stair riser to a fresh fracture of the same stone. The reduction in hardness correlates with exposure time (chemical weathering).
   - *Reliability:* Moderate-High. Independent of traffic flux.

### 6.3 Guidance: Detecting Repairs and Renovations

**Mathematical Signature:**

Renovations introduce discontinuities in the wear function.

- **Type 1: Step Replacement.** A single step has significantly less wear ($h \ll h_{neighbors}$) or a different wear profile shape (e.g., U-shaped vs V-shaped), indicating a different material hardness or a "reset" clock.
- **Type 2: Handrail Installation.** A sudden shift in the lateral mean $\mu$ of the wear track. If the lower stairs show $\mu = 0.5m$ (from wall) and upper stairs show $\mu = 0.3m$, a handrail or obstacle may have been added/removed at some point in history, altering the effective width.
- **Type 3: Re-facing (Flipping).** Sometimes stone slabs are flipped over. Look for wear on the *underside* of the tread using an endoscope or during excavation.

### 6.4 Guidance: Determining Material Provenance

**Technique: Portable X-Ray Fluorescence (pXRF)**

Visual inspection is insufficient for determining if a replacement stone came from the original quarry.

1. **Protocol:** Take pXRF readings of the original stones (Cluster A) and the suspected repairs (Cluster B).
2. **Metrics:** Analyze trace element ratios (e.g., Ti/Zr, Rb/Sr, Fe/Mn).
3. **Analysis:** If Cluster B falls outside the $95\%$ confidence ellipse of Cluster A in compositional space, the material is from a different geological source (quarry), confirming a later intervention or supply chain shift.

------

## 7. Field Methodology: A Guide for the Archaeologist

We define a non-destructive, low-cost data collection protocol required to feed the mathematical model.

### 7.1 Required Measurements and Tools

| **Measurement**        | **Variable in Model**          | **Tool**                       | **Protocol**                                                 |
| ---------------------- | ------------------------------ | ------------------------------ | ------------------------------------------------------------ |
| **Step Geometry**      | $W, G, R$ (Width, Going, Rise) | Laser Distometer / Tape        | Measure every step to detect variances in construction.      |
| **Wear Depth Profile** | $h(x,y)$                       | **Photogrammetry**             | Take overlapping photos (60%) with a scale bar. Use SfM software to generate a DEM. Subtract the "unworn edge" plane from the mesh to isolate wear depth. |
| **Surface Hardness**   | $H$                            | **Schmidt Hammer (Type N/L)**  | Take 10 impacts on the wear track and 10 on the unworn edge. The difference calibrates the "wearability" and detects weathering crusts. |
| **Material Comp.**     | $k_{spec}$ (proxy)             | **pXRF**                       | Analyze trace elements to group stones by quarry source.     |
| **Roughness**          | $\gamma_{env}$                 | **Profile Gauge / Comparator** | Measure roughness ($R_a$) on unworn edges to estimate environmental weathering age. |



### 7.2 Data Processing Workflow

1. **Generate 3D Model:** Create a high-resolution Digital Elevation Model (DEM) of the staircase.
2. **Extract Profiles:** Extract cross-sectional profiles (cuts along the $x$-axis) for each tread.
3. **Compute Moments:** For each profile, calculate:
   - **Mean ($\mu$):** Center of traffic.
   - **Standard Deviation ($\sigma$):** Spread of traffic.
   - **Kurtosis ($K$):** Peakedness (Single file vs. distributed).
   - **Skewness ($S$):** Asymmetry (Wall bias).
4. **Cluster Steps:** Use K-Means clustering on these moments. Steps that do not cluster with the main group are likely **replacements** or **renovations**.

------

## 8. Case Study Simulation

**Scenario:** An archaeologist investigates a stone staircase in a medieval tower.

- **Observation:** Steps 1-10 show a deep, single central groove. Steps 11-20 show a wider, flatter wear pattern.
- **Measurement:**
  - Steps 1-10: Width = 0.9m. Kurtosis = 4.2 (Leptokurtic).
  - Steps 11-20: Width = 1.5m. Kurtosis = 2.5 (Platykurtic).
- **Model Interpretation:**
  - The change in wear profile is driven by **geometry**, not a change in usage volume. The narrow lower stairs forced single-file traffic (high kurtosis). The wider upper stairs allowed lateral dispersion (low kurtosis).
  - *Correction:* When calculating total traffic volume $N_{total}$, the archaeologist must use a different **Area Correction Factor** for the two sections. The wear volume in the narrow section will be *deeper* for the same number of people. Using simple depth would overestimate traffic on the lower stairs. The model normalizes this by integrating volume, not just depth.

**Scenario:** Granite steps ($k_{spec} \approx 10^{-6}$) show 20mm of wear.

- **Calculation:**

  $$N_{steps} \approx \frac{V_{wear}}{k_{spec} \cdot W_{load} \cdot L_{slide}}$$

  Assuming average load (700N) and microslip (2mm/step):

  $$N \approx \frac{0.0002 m^3}{10^{-15} \cdot 700 \cdot 0.002} \approx 1.4 \times 10^8 \text{ steps}$$

- **Conclusion:** 140 million steps. If the building is 500 years old, this implies $\approx 760$ people per day, every day. If the historical record suggests only 10 monks lived there, the age estimate is wrong (structure is older) or the stone is softer than standard granite (requires calibration via Schmidt hammer).

------

## 9. Conclusion

The wear on a staircase is a deterministic function of material properties and human kinetics, overlaid with a stochastic distribution of foot placement. By modeling the step as a grid and applying Archard’s equation differentially across the surface, archaeologists can effectively "reverse engineer" the history of the structure.

**Key Recommendations:**

1. **Measure the Edges:** The unworn edge is the baseline for "zero wear" and the clock for "background weathering."
2. **Look at the Nosing:** The curvature of the step edge is the primary indicator of descent traffic.
3. **Trust the Variance:** A bimodal wear pattern is the strongest evidence for high-volume, simultaneous bidirectional flow.
4. **Chemistry over Visuals:** Use pXRF to identify repairs that look visually identical but have different mineral compositions.

This mathematical approach transforms the worn stone from a mere romantic symbol of age into a rigorous dataset, capable of revealing the daily lives and movements of the people who traversed it centuries ago.

------

### **Appendix: Mathematical Derivation of Gaussian Wear Profile**

The lateral wear profile $W(x)$ is modeled as the convolution of the foot pressure kernel $P(x)$ and the pedestrian lateral distribution $D(x)$:

$$W(x) = \int_{-\infty}^{\infty} D(x') P(x - x') dx'$$

Assuming $D(x)$ is Gaussian $\mathcal{N}(\mu, \sigma_D^2)$ and $P(x)$ is Gaussian $\mathcal{N}(0, \sigma_P^2)$ (approximating the foot width):

$$W(x) \propto \mathcal{N}(\mu, \sigma_D^2 + \sigma_P^2)$$

Taking the natural log of the wear profile:

$$\ln(W(x)) \propto -\frac{(x-\mu)^2}{2(\sigma_D^2 + \sigma_P^2)}$$

Plotting $\ln(W(x))$ vs. $x$ should yield a parabola. Deviations from this parabola (e.g., flat tops, heavy tails) indicate:

- **Non-Gaussian Behavior:** Constrained flow (walls) or "Lane Formation" (superposition of two parabolas).
- **Changes in $\mu$ (Lane Shift):** Renovation or obstacle placement.

This simple log-plot test can be performed in the field to check for traffic stability over time.