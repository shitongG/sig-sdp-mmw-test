# Harmony Deterministic Scheduling Website Design

Date: 2026-03-28

## 1. Goal

Build a single static presentation website, visually styled like a formal thesis-defense PPT, to introduce the current research and experimental progress on deterministic wireless communication scheduling for HarmonyOS distributed collaboration scenarios.

The page is intended for advisor/reviewer presentation use, not for consumer marketing. It should read like a polished research briefing: clear problem statement, explicit modeling assumptions, technically accurate algorithm explanation, and credible experiment status.

The website must cover four required content pillars:

1. Research scenario
2. Current modeling rules
3. Difficulties of existing methods
4. Benefits of the chosen scheduling approach

In addition, the user explicitly requires the page to clearly explain:

1. Current experimental progress
2. Current experimental results
3. Concrete algorithm principles

## 2. Delivery Format

This will be implemented as a directly openable static website:

- `site/index.html`
- `site/styles.css`
- `site/script.js`

No build system is required.

The page must work by opening `index.html` directly in a browser.

## 3. Audience and Tone

Audience:

- advisor
- reviewers
- thesis-defense style evaluators

Tone:

- formal
- technical
- high-credibility
- presentation-oriented

This should look like a strong defense deck converted into a scrollable webpage, not like a startup landing page.

## 4. Visual Direction

Recommended direction: defense-narrative single-page presentation.

Characteristics:

- dark graphite background
- restrained, high-contrast academic palette
- cold white body text
- cyan-blue as the primary technical accent
- amber/orange for constraints, conflicts, and challenge emphasis
- large section transitions that feel like PPT slides
- subtle motion only for reveal and emphasis, not for decoration

The memorable visual identity should be:

"A deterministic scheduling defense board: dense enough for technical trust, clean enough for formal presentation."

## 5. Information Architecture

The page should be structured as a long-form, slide-like narrative with 6 sections.

### 5.1 Hero / Cover

Purpose:

- establish the problem domain immediately
- frame the page as a research presentation

Content:

- title around HarmonyOS distributed collaborative deterministic wireless communication
- subtitle explaining this is a task-driven WiFi/BLE joint scheduling study
- short one-sentence problem statement:
  - multi-device collaborative tasks contend for wireless airtime
  - random wireless contention harms deterministic experience

Visuals:

- strong title block
- a thin signal-spectrum motif
- a side annotation block with keywords:
  - deterministic
  - distributed collaboration
  - WiFi/BLE joint scheduling

### 5.2 Research Scenario

Purpose:

- explain the real application scene and why the problem matters

Core source content:

- HarmonyOS distributed collaboration
- WPS calling phone camera from PC and directly inserting image into the PC document
- multi-device, multi-task, concurrent wireless communication

Presentation:

- show a concrete workflow:
  - PC requests phone camera
  - phone captures image
  - image/task fragments transmitted back
  - content inserted into WPS document
- optionally show other collaboration examples in smaller supporting cards:
  - multi-PC collaborative editing
  - clipboard / device continuation / service interconnection

Required emphasis:

- devices are mobile and power constrained
- wireless medium is shared and collision-prone
- multiple concurrent tasks create contention

### 5.3 Modeling Rules

Purpose:

- make the current simulation abstraction explicit

This section must clearly answer:

- what is a task
- when is a task mapped to BLE
- when is a task mapped to WiFi
- what traffic types are assumed for WiFi and BLE
- what is jointly scheduled

Required content:

1. Workflow-to-task abstraction
   - the WPS camera insertion workflow is decomposed into several communication tasks
   - each task has data size and deadline

2. Radio selection rule
   - each task is mapped to BLE or WiFi according to transmission demand
   - low-volume / timing-sensitive tasks can be abstracted to BLE-style transmission
   - higher-volume tasks can be abstracted to WiFi-style transmission

3. Traffic model
   - WiFi traffic is modeled as periodic occupancy with wider bandwidth
   - BLE traffic is modeled as periodic connect events / hopping events with narrow bandwidth

4. Scheduling abstraction
   - scheduling is performed over time-frequency resource blocks
   - WiFi and BLE are jointly placed under conflict-free constraints

5. Constraints to mention explicitly
   - release time / deadline
   - periodicity
   - channel occupancy width
   - BLE data-channel constraints and broadcast-channel avoidance
   - WiFi channel restrictions

This section should include one clean pipeline diagram:

`application workflow -> tasks -> radio selection -> task pairs -> joint scheduling`

### 5.4 Difficulties of Existing Methods

Purpose:

- explain why this research direction is needed

Content should summarize:

- random backoff and probabilistic anti-collision mechanisms are insufficient for deterministic communication
- existing deterministic solutions are largely wired-network oriented
- device-side coarse traffic suppression is too crude
- device-to-device airtime contention remains unordered
- dense, dynamic, power-constrained multi-device environments make centralized methods difficult

The layout should visually separate difficulties into 3 layers:

1. Protocol-layer difficulty
2. Scheduling-layer difficulty
3. System-level difficulty

Suggested bullets:

- QoS objectives conflict: throughput, delay, fairness, power
- global airtime utilization is hard under decentralized, dynamic wireless access
- static or probabilistic strategies cannot guarantee predictable completion
- naive coexistence causes collisions and retreat overhead

### 5.5 Method and Algorithm Principle

Purpose:

- explain the core technical contribution in a formal, research-oriented way

This is the most technical section and must be more detailed than the previous four sections.

Required subsections:

1. Why joint scheduling
   - scheduling WiFi and BLE independently causes mutual blocking
   - task-driven joint scheduling can better allocate scarce wireless resources

2. Current algorithm line
   - BLE-only SDP
   - BLE-only GA
   - WiFi-first plus BLE scheduling
   - iterative WiFi-BLE coordination
   - unified joint GA / HGA exploration

3. Current unified scheduling idea
   - represent WiFi and BLE candidate states under one joint state space
   - optimize under conflict-free time-frequency constraints
   - preserve WiFi baseline quality while improving BLE packing where possible

4. Algorithm explanation
   - candidate-state construction
   - chromosome / encoding idea for GA and HGA
   - fitness / objective priorities
   - residual-hole / packing intuition

This section does not need full mathematical derivation on the page, but it must include technically meaningful explanations instead of only high-level slogans.

### 5.6 Experimental Progress and Current Results

Purpose:

- show that this is an actively advancing simulation study, not just a concept

This section should explicitly summarize current progress:

- BLE macrocycle hopping SDP implemented
- BLE macrocycle hopping GA implemented
- WiFi-first plus BLE scheduling implemented
- iterative WiFi-BLE coordination explored
- unified joint GA/HGA explored in isolated experimental branch

It must also explain the current status honestly:

- some directions improve total joint scheduling quantity
- some directions preserve WiFi payload better
- the unified joint HGA is promising but has not yet consistently surpassed the earlier WiFi-first heuristic in all metrics

Recommended display:

- one timeline or milestone band
- one comparison table with compact metrics
- one “current conclusion” callout

This section should present results as interim experimental evidence, not final claims.

## 6. Content Style Rules

The page text should:

- be written in Chinese
- use concise but formal language
- avoid colloquial marketing language
- keep terminology consistent:
  - task
  - deadline
  - BLE / WiFi
  - joint scheduling
  - time-frequency resource blocks
  - deterministic communication

The page should not overuse formulas. It should prioritize:

- diagrams
- structured bullets
- concise technical captions

## 7. Interaction and Motion

The page should include restrained presentation-style interactions:

- fade / slide reveal on scroll
- staggered appearance for metric cards
- subtle highlight motion for process arrows and spectrum bars

Do not add:

- flashy particle systems
- excessive parallax
- distracting hover gimmicks

## 8. Accessibility and Layout

Requirements:

- desktop-first because the page is for formal review
- still readable on laptop and tablet
- responsive stacking on narrow screens
- strong contrast
- legible body text

## 9. Files to Create

- `site/index.html`
- `site/styles.css`
- `site/script.js`

Optional:

- `site/assets/` only if needed for icons or small decorative assets

## 10. Non-Goals

This page is not:

- a full documentation portal
- an interactive simulator frontend
- a general project homepage

It is a presentation website for communicating the current research and experiment status.

## 11. Recommended Implementation Approach

Use a pure static implementation:

- semantic HTML sections
- custom CSS for the presentation aesthetic
- light JavaScript only for reveal effects and section navigation polish

This is the lowest-risk way to produce a polished, portable presentation page quickly.

## 12. Success Criteria

The page is successful if:

1. A reviewer can understand the research problem in under 1 minute
2. A reviewer can understand the current modeling assumptions without opening the code
3. A reviewer can understand why WiFi/BLE joint scheduling is needed
4. A reviewer can see what has been implemented already and what remains open
5. The page looks formal, distinctive, and defense-ready rather than template-like
