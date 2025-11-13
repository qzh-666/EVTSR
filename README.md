# EvTSR: Event-Guided Scene Text Image Super-Resolution

This repository contains the official open-source code for our work, **"Event-Guided Scene Text Image Super-Resolution"** (**AAAI-26**).

## 📜 Abstract

Scene text image super-resolution aims to enhance text legibility by recovering high-resolution text images from low-resolution inputs. However, maintaining fine details such as text strokes, edges, and textual accuracy remains challenging, particularly in low-light environments and high-speed motion scenarios, where degradation is more severe. Event cameras, with their high temporal resolution and ability to capture intensity changes, offer a promising solution for restoring lost fine details and mitigating degradation in these challenging conditions. In this paper, we propose EvTSR, the first framework that integrates Event data for scene Text image Super-Resolution. The core of EvTSR is the dual stream frequency boost (DSFB) mechanism, which separates image features into high- and low-frequency components. High-frequency details like edges and strokes are enhanced using event data via the event-guided high-frequency (EGH) module, while low-frequency components, responsible for global structure, are refined using the Text-Guided Low-frequency (TGL) module with a pre-trained text recognizer, ensuring textual coherence. To further improve cross-modal integration, we introduce the Cross-Modal Fusion (CMF) module, which effectively aligns event and image features, enabling robust information fusion. Extensive experiments demonstrate that EvTSR achieves superior performance over existing methods.

## 🖼️ Framework Overview
<img width="2426" height="669" alt="image" src="https://github.com/user-attachments/assets/f84dc668-3f8b-4ba0-9b57-1c19a769923c" />

## 🙏 Acknowledgements
Our training framework is based on the official implementation of [EvTexture (ICML 2024)](https://github.com/DachunKai/EvTexture). We primarily utilize their training and testing pipeline, replacing the model architecture (`arch`) code with our proposed EvTSR network.
