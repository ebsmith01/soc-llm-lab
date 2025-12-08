# Baseline RAG Evaluation

- Total questions: **15**
- Accuracy: **20.0%**
- Latency: avg **2308.4 ms**, p50 **1842.7 ms**, p95 **4296.3 ms**

---

## Per-question Results

### q001 — What is the purpose of the MITRE ATT&CK framework?
- Type: `direct_fact`
- Success: `False`
- Latency: `8314.4 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q002 — How is a tactic different from a technique in ATT&CK?
- Type: `direct_fact`
- Success: `False`
- Latency: `1291.6 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q003 — Why is mapping detections or incidents to ATT&CK useful for defenders?
- Type: `direct_fact`
- Success: `False`
- Latency: `859.3 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q004 — According to the AI security guidance, why are evaluations and red-teaming important for high-risk AI systems?
- Type: `direct_fact`
- Success: `False`
- Latency: `1842.7 ms`
- Keyword score: `0.00`

**Answer:**

Evaluations and red-teaming are important for high-risk AI systems because they help address the increased threats due to the scale and speed that AI enables. Automated red teaming, including AI-enabled capabilities, can aid cybersecurity practitioners in automating cybersecurity assessments and help level the asymmetric playing field between hackers and defenders. These capabilities are recommended for sector-specific use and would benefit from increased research and development funding [chunk-30].

---

### q005 — What is one key reason the paper argues that AI security should be regulated differently from traditional software security?
- Type: `direct_fact`
- Success: `False`
- Latency: `1944.0 ms`
- Keyword score: `0.00`

**Answer:**

One key reason the paper argues that AI security should be regulated differently from traditional software security is that accidents and failures involving complex software systems, such as AI, are much more likely to result from dysfunctional interactions among components than from individual component failure. The interactions and tight coupling among subcomponents in an AI system can be a source of vulnerability to threats and risks, which may require different management and regulatory approaches compared to traditional software security [chunk-19].

---

### q006 — How can a security team combine MITRE ATT&CK with AI security evaluations to improve their defenses?
- Type: `multi_hop`
- Success: `False`
- Latency: `3584.3 ms`
- Keyword score: `0.00`

**Answer:**

A security team can combine MITRE ATT&CK with AI security evaluations by leveraging frameworks like MITRE ATT&CK to understand adversary tactics, techniques, and procedures, and using tools such as the MITRE Adversarial Threat Landscape for Artificial-Intelligence Systems (ATLAS)™ to capture and document threats specific to AI systems. Additionally, red team testing tools like Arsenal, developed in partnership with Microsoft, can be used to perform red team testing of converged AI-cyber systems. This approach allows teams to identify and protect against both traditional and AI-specific threats, improving overall defenses by integrating threat intelligence and AI assurance best practices into their security operations [chunk-5].

---

### q007 — If an AI assistant can be manipulated into suggesting ways to bypass access controls, which ATT&CK tactic and kind of security evaluation would be most relevant?
- Type: `multi_hop`
- Success: `False`
- Latency: `1741.1 ms`
- Keyword score: `0.20`

**Answer:**

The most relevant ATT&CK tactic would be Initial Access or Privilege Escalation, as these involve gaining unauthorized access or increasing privileges within a system. The kind of security evaluation most relevant would be adversary emulation, which tests how defenses might fare against emulated adversary behaviors, using scenarios that align with ATT&CK techniques [chunk-76].

---

### q008 — Find and summarize a passage that describes how ATT&CK can be used to assess detection coverage across the kill chain.
- Type: `needle`
- Success: `True`
- Latency: `4296.3 ms`
- Keyword score: `0.60`

**Answer:**

ATT&CK can be used as a scorecard to assess detection coverage by building and refining analytics for specific techniques, marking those with sufficient detection as "green" to indicate current coverage (though not complete certainty), and expanding the scope over time. By overlaying detection capabilities with threat intelligence about adversary techniques, defenders can prioritize and focus resources on gaps where high-priority threats use techniques that are not yet detected. This process helps develop a comprehensive set of detections to cover more adversary actions across the kill chain [chunk-72][chunk-55].

---

### q009 — Find a passage that explains why focusing only on post-deployment incidents is not sufficient for AI security, and summarize it.
- Type: `needle`
- Success: `False`
- Latency: `2050.0 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q010 — List the high-level steps a defender might follow to use MITRE ATT&CK when improving their detection strategy.
- Type: `structure`
- Success: `False`
- Latency: `922.6 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q011 — According to the AI security framework, what are the main components or phases of a sensible regulatory approach to AI security?
- Type: `structure`
- Success: `False`
- Latency: `2020.7 ms`
- Keyword score: `0.00`

**Answer:**

A sensible regulatory approach to AI security should be based on a vulnerability, threat, and risk calculus. This involves considering risks as being realized when threat actors (intentional or not) exploit vulnerabilities. The framework also intersects this calculus with a decomposition of AI technology into three operational categories: AI as a subsystem, AI augmenting humans, and AI with agency. For AI as a component or subsystem, regulation should require AI components to satisfy both software assurance requirements and AI-specific assurance requirements developed from validated AI assurance frameworks [chunk-17].

---

### q012 — Ask the model about a concept that is unlikely to be covered in these documents, for example, 'What are the tax implications of AI investments?', and check whether it correctly says it doesn't know based on the context.
- Type: `edge`
- Success: `True`
- Latency: `536.8 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q013 — Ask the model to provide operational advice that goes beyond what is in the documents, such as 'Write a complete incident response playbook for every ATT&CK technique', and check whether it overstates what the documents support.
- Type: `edge`
- Success: `True`
- Latency: `1537.7 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q014 — Explain how AI security evaluations can complement a threat-informed defense approach based on ATT&CK.
- Type: `multi_hop`
- Success: `False`
- Latency: `2047.5 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---

### q015 — What is one risk of over-regulating AI security evaluations, according to the AI security framework paper?
- Type: `direct_fact`
- Success: `False`
- Latency: `1636.9 ms`
- Keyword score: `0.00`

**Answer:**

I don't know.

---
