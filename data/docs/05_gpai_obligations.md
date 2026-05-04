# EU AI Act — General-Purpose AI (GPAI) Models

Chapter V (Articles 51–56) introduces a regime for general-purpose AI models, distinct from the system-level rules. A GPAI model is defined as an AI model trained on a large amount of data using self-supervision at scale, displaying significant generality, and capable of competently performing a wide range of distinct tasks regardless of how it is placed on the market. Large language models, foundation image and video models, and similar models are in scope.

## Two tiers

The regime distinguishes between (i) all GPAI models and (ii) GPAI models with systemic risk.

### All GPAI providers

Article 53 obligations:

- Draw up and keep up-to-date technical documentation of the model, including its training and testing process and the results of its evaluation, to be provided to the AI Office and national competent authorities on request (Annex XI).
- Make information and documentation available to downstream providers who intend to integrate the model into their AI systems, sufficient for them to understand capabilities and limitations and comply with their own obligations (Annex XII).
- Put in place a policy to comply with Union copyright law, in particular to identify and respect — including through state-of-the-art technologies — reservations of rights expressed under Article 4(3) of Directive (EU) 2019/790 (the text-and-data-mining opt-out).
- Draw up and make publicly available a sufficiently detailed summary of the content used for training the model, according to a template provided by the AI Office.

Open-source GPAI models (released under a free and open licence allowing access, use, modification and distribution, with weights and architecture publicly available) are exempted from the documentation duties and the duty to provide information to downstream providers, unless they are systemic-risk models. The copyright-policy and training-data-summary duties still apply.

### Systemic-risk GPAI

A GPAI model is presumed to pose systemic risk when it has high-impact capabilities, with the cumulative compute used for training measured in floating-point operations greater than 10^25 FLOPs (Article 51). The AI Office may also designate models as systemic-risk on the basis of equivalent capabilities or impact, applying criteria in Annex XIII (number of parameters, dataset quality, modalities, reach, etc.).

Providers of systemic-risk GPAI models bear additional obligations under Article 55:

- Perform model evaluation in accordance with standardised protocols and tools reflecting the state of the art, including conducting and documenting adversarial testing of the model with a view to identifying and mitigating systemic risks.
- Assess and mitigate possible systemic risks at Union level, including their sources, that may stem from the development, placing on the market, or use of the model.
- Track, document and report serious incidents and possible corrective measures to the AI Office and, as appropriate, to national competent authorities, without undue delay.
- Ensure an adequate level of cybersecurity protection for the model and its physical infrastructure.

## Codes of practice and harmonised standards

Article 56 invites providers to participate in drawing up Codes of Practice covering documentation, copyright, systemic-risk assessment and mitigation, evaluation, and incident reporting. The first Codes are coordinated by the AI Office and adherence is one route to demonstrating compliance, alongside formal harmonised standards once they are published. Pre-existing GPAI models placed on the market before 2 August 2025 have until 2 August 2027 to come into conformity.
