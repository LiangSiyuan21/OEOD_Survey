# OEOD_Survey
This is the repository of **Vision Language Models for Vision Tasks: a Survey**, a systematic survey of VLM studies in various visual recognition tasks including image classification, object detection, semantic segmentation, etc. For details, please refer to:

**Vision-Language Models for Vision Tasks: A Survey**  
 [[Paper](https://arxiv.org/abs/2304.00685)]
 
[![arXiv](https://img.shields.io/badge/arXiv-2304.00685-b31b1b.svg)](https://arxiv.org/abs/2304.00685) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) 
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
<!-- [![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest) -->

*Feel free to contact us or pull requests if you find any related papers that are not included here.*


## Abstract

Most visual recognition studies rely heavily on crowd-labelled data in deep neural networks (DNNs) training, and they usually train a DNN for each single visual recognition task, leading to a laborious and time-consuming visual recognition paradigm. To address the two challenges, Vision Language Models (VLMs) have been intensively investigated recently, which learns rich vision-language correlation from web-scale image-text pairs that are almost infinitely available on the Internet and enables zero-shot predictions on various visual recognition tasks with a single VLM. This paper provides a systematic review of visual language models for various visual recognition tasks, including: (1) the background that introduces the development of visual recognition paradigms; (2) the foundations of VLM that summarize the widely-adopted network architectures, pre-training objectives, and downstream tasks; (3) the widely adopted datasets in VLM pre-training and evaluations; (4) the review and categorization of existing VLM pre-training methods, VLM transfer learning methods, and VLM knowledge distillation methods; (5) the benchmarking, analysis and discussion of the reviewed methods; (6) several research challenges and potential research directions that could be pursued in the future VLM studies for visual recognition.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2023vision,
  title={Vision-Language Models for Vision Tasks: A Survey},
  author={Zhang, Jingyi and Huang, Jiaxing and Jin, Sheng and Lu, Shijian},
  journal={arXiv preprint arXiv:2304.00685},
  year={2023}
}
```

## Menu
- [Vision-Language Pre-training Methods](#Out-of-domain-benchmark)
  - [Pre-training with Contrastive Objective](#pre-training-with-contrastive-objective)
  - [Pre-training with Generative Objective](#pre-training-with-generative-objective)
  - [Pre-training with Alignment Objective](#pre-training-with-alignment-objective)
- [Vision-Language Model Transfer Learning Methods](#vision-language-model-transfer-learning-methods)
  - [Transfer with Prompt Tuning](#transfer-with-prompt-tuning)
    - [Transfer with Text Prompt Tuning](#transfer-with-text-prompt-tuning)
    - [Transfer with Visual Prompt Tuning](#transfer-with-visual-prompt-tuning)
    - [Transfer with Text and Visual Prompt Tuning](#transfer-with-text-and-visual-prompt-tuning)
  - [Transfer with Feature Adapter](#transfer-with-feature-adapter)
  - [Transfer with Other Methods](#transfer-with-other-methods)
- [Vision-Language Model Knowledge Distillation Methods](#vision-language-model-knowledge-distillation-methods)
  - [Knowledge Distillation for Object Detection](#knowledge-distillation-for-object-detection)
  - [Knowledge Distillation for Semantic Segmentation](#knowledge-distillation-for-semantic-segmentation)

## Out of domain benchmark

### Data manipulation

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Diversify and Match: A Domain Adaptive Representation Learning Paradigmfor Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.html)|CVPR 2019|[Code](https://github.com/TKKim93/DivMatch)|
|[Domain randomization for scene-specific car detection and pose estimation](https://ieeexplore.ieee.org/abstract/document/8658387)|WACV 2019|-|
|[Structured domain randomization: Bridging the reality gap by context-aware synthetic data](https://ieeexplore.ieee.org/abstract/document/8794443)|ICRA 2019|-|
|[AFAN: augmented feature alignment network for cross-domain object detection](https://ieeexplore.ieee.org/abstract/document/9393610)|TIP 2021|-|
|[Progressive domain adaptation for object detection](https://openaccess.thecvf.com/content_WACV_2020/html/Hsu_Progressive_Domain_Adaptation_for_Object_Detection_WACV_2020_paper.html)|WACV 2020|[Code](https://github.com/kevinhkhsu/DA_detection)|
|[Harmonizing transferability discriminability for adapting object detectors](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Harmonizing_Transferability_and_Discriminability_for_Adapting_Object_Detectors_CVPR_2020_paper.html)|CVPR 2020|[Code](https://github.com/chaoqichen/HTCN)|
|[Unpaired Image-To-Image Translation Using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)|ICCV 2017|[Code](https://github.com/hanyoseob/pytorch-CycleGAN)|
|[AsyFOD: An asymmetric adaptation paradigm for few-shot domain adaptive object detection](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_AsyFOD_An_Asymmetric_Adaptation_Paradigm_for_Few-Shot_Domain_Adaptive_Object_CVPR_2023_paper.html)|CVPR 2023|[Code](https://github.com/Hlings/AsyFOD)|
|[CLIP the gap: A single domain generalization approach for object detection](http://openaccess.thecvf.com/content/CVPR2023/html/Vidit_CLIP_the_Gap_A_Single_Domain_Generalization_Approach_for_Object_CVPR_2023_paper.html)|CVPR 2023|-|
|[Auggan: Cross domain adaptation with gan-based data augmentation](http://openaccess.thecvf.com/content_ECCV_2018/html/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.html)|ECCV 2018|-|
|[Domain adaptation for object detection via style consistency](https://arxiv.org/abs/1911.10033)|BMVC 2019|-|
|[Target-style-aware unsupervised domain adaptation for object detection](https://ieeexplore.ieee.org/abstract/document/9363588/)|RAL 2021|-|
|[SC-UDA: style content gaps aware unsupervised domain adaptation for object detection](https://openaccess.thecvf.com/content/WACV2022/html/Yu_SC-UDA_Style_and_Content_Gaps_Aware_Unsupervised_Domain_Adaptation_for_WACV_2022_paper.html)|WACV 2022|-|




### Pre-training with Generative Objective

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482)|CVPR 2022|[Code](https://github.com/facebookresearch/multimodal/tree/main/examples/flava)|
|[CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)|arXiv 2022|[Code](https://github.com/lucidrains/CoCa-pytorch)|
|[Too Large; Data Reduction for Vision-Language Pre-Training](https://arxiv.org/abs/2305.20087)|arXiv 2023|[Code](https://github.com/showlab/data-centric.vlp)|
|[SAM: Segment Anything](https://arxiv.org/abs/2304.02643)|arXiv 2023|[Code](https://github.com/facebookresearch/segment-anything)|
|[SEEM: Segment Everything Everywhere All at Once](https://arxiv.org/pdf/2304.06718.pdf)|arXiv 2023|[Code](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)|
|[Semantic-SAM: Segment and Recognize Anything at Any Granularity](https://arxiv.org/pdf/2307.04767.pdf)|arXiv 2023|[Code](https://github.com/UX-Decoder/Semantic-SAM)|



### Pre-training with Alignment Objective

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)|CVPR 2022|[Code](https://github.com/microsoft/GLIP)|
|[DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection](https://arxiv.org/abs/2209.09407)|NeurIPS 2022|-|
|[nCLIP: Non-Contrastive Learning Meets Language-Image Pre-Training](https://arxiv.org/abs/2210.09304)|CVPR 2023|[Code](https://github.com/shallowtoil/xclip)|

## Vision-Language Model Transfer Learning Methods

### Transfer with Prompt Tuning

#### Transfer with Text Prompt Tuning

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)|IJCV 2022|[Code](https://github.com/KaiyangZhou/CoOp)|
|[CoCoOp: Conditional Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2203.05557)|CVPR 2022|[Code](https://github.com/KaiyangZhou/CoOp)|
|[ProDA: Prompt Distribution Learning](https://arxiv.org/abs/2205.03340)|CVPR 2022|-|
|[DenseClip: Language-Guided Dense Prediction with Context-Aware Prompting](https://arxiv.org/abs/2112.01518)|CVPR 2022|[Code](https://github.com/raoyongming/DenseCLIP)|
|[TPT: Test-time prompt tuning for zero-shot generalization in vision-language models](https://arxiv.org/abs/2209.07511)|NeurIPS 2022|[Code](https://github.com/azshue/TPT)|
|[DualCoOp: Fast Adaptation to Multi-Label Recognition with Limited Annotations](https://arxiv.org/abs/2206.09541)|NeurIPS 2022|[Code](https://github.com/sunxm2357/DualCoOp)|
|[CPL: Counterfactual Prompt Learning for Vision and Language Models](https://arxiv.org/abs/2210.10362)|EMNLP 2022|[Code](https://github.com/eric-ai-lab/CPL)|
|[Bayesian Prompt Learning for Image-Language Model Generalization](https://arxiv.org/abs/2210.02390v2)|arXiv 2022|-|
|[UPL: Unsupervised Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2204.03649)|arXiv 2022|[Code](https://github.com/tonyhuang2022/UPL)|
|[ProGrad: Prompt-aligned Gradient for Prompt Tuning](https://arxiv.org/abs/2205.14865)|arXiv 2022|[Code](https://github.com/BeierZhu/Prompt-align)|
|[SoftCPT: Prompt Tuning with Soft Context Sharing for Vision-Language Models](https://arxiv.org/abs/2208.13474)|arXiv 2022|[Code](https://github.com/kding1225/softcpt)|
|[SubPT: Understanding and Mitigating Overfitting in Prompt Tuning for Vision-Language Models](https://arxiv.org/abs/2211.02219)|TCSVT 2023|[Code](https://github.com/machengcheng2016/Subspace-Prompt-Learning)|
|[LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models](https://arxiv.org/abs/2210.01115)|CVPR 2023|[Code](https://www.adrianbulat.com/lasp)|
|[PLOT: Prompt Learning with Optimal Transport for Vision-Language Models](https://arxiv.org/abs/2210.01253)|ICLR 2023|[Code](https://github.com/CHENGY12/PLOT)|
|[LMPT: Prompt Tuning with Class-Specific Embedding Loss for Long-tailed Multi-Label Visual Recognition](https://arxiv.org/abs/2305.04536)|arXiv 2023|[Code](https://github.com/richard-peng-xia/LMPT)|
|[Texts as Images in Prompt Tuning for Multi-Label Image Recognition](https://arxiv.org/abs/2211.12739)|CVPR 2023|[code](https://github.com/guozix/TaI-DPT)
|[Visual-Language Prompt Tuning with Knowledge-guided Context Optimization](https://arxiv.org/abs/2303.13283)|CVPR 2023|[Code](https://github.com/htyao89/KgCoOp)|
|[Learning to Name Classes for Vision and Language Models](https://arxiv.org/abs/2304.01830v1)|CVPR 2023|-|
|[CuPL: What does a platypus look like? Generating customized prompts for zero-shot image classification](https://arxiv.org/abs/2209.03320)|ICCV 2023|[Code](https://github.com/sarahpratt/CuPL)|
|[ProTeCt: Prompt Tuning for Hierarchical Consistency](https://arxiv.org/abs/2306.02240)|arXiv 2023|-|
|[Enhancing CLIP with CLIP: Exploring Pseudolabeling for Limited-Label Prompt Tuning](https://arxiv.org/abs/2306.01669)|arXiv 2023|[Code](http://github.com/BatsResearch/menghini-enhanceCLIPwithCLIP-code)|


#### Transfer with Visual Prompt Tuning

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Exploring Visual Prompts for Adapting Large-Scale Models](https://arxiv.org/abs/2203.17274)|arXiv 2022|[Code](https://github.com/hjbahng/visual_prompting)|
|[Retrieval-Enhanced Visual Prompt Learning for Few-shot Classification](https://arxiv.org/abs/2306.02243)|arXiv 2023|-|
|[Fine-Grained Visual Prompting](https://arxiv.org/abs/2306.04356)|arXiv 2023|-|

#### Transfer with Text and Visual Prompt Tuning

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[UPT: Unified Vision and Language Prompt Learning](https://arxiv.org/abs/2210.07225)|arXiv 2022|[Code](https://github.com/yuhangzang/upt)|
|[MVLPT: Multitask Vision-Language Prompt Tuning](https://arxiv.org/abs/2211.11720)|arXiv 2022|[Code](https://github.com/facebookresearch/vilbert-multi-task)|
|[CAVPT: Dual Modality Prompt Tuning for Vision-Language Pre-Trained Model](https://arxiv.org/abs/2208.08340)|arXiv 2022|[Code](https://github.com/fanrena/DPT)|
|[MaPLe: Multi-modal Prompt Learning](https://arxiv.org/abs/2210.03117)|CVPR 2023|[Code](https://github.com/muzairkhattak/multimodal-prompt-learning)|

### Transfer with Feature Adapter

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Tip-Adapte: Training-free Adaption of CLIP for Few-shot Classification](https://arxiv.org/abs/2207.09519)|ECCV 2022|[Code](https://github.com/gaopengcuhk/Tip-Adapter)|
|[SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models](https://arxiv.org/abs/2210.03794)|BMVC 2022|[Code](https://github.com/omipan/svl_adapter)|
|[Clip-Adapter: Better Vision-Language Models with Feature Adapters](https://arxiv.org/abs/2110.04544)|arXiv 2021|[Code](https://github.com/gaopengcuhk/CLIP-Adapter)|
|[SuS-X: Training-Free Name-Only Transfer of Vision-Language Models](https://arxiv.org/abs/2211.16198)|ICCV 2023|[Code](https://github.com/vishaal27/SuS-X)|
|[CLIPPR: Improving Zero-Shot Models with Label Distribution Priors](https://arxiv.org/abs/2212.00784)|arXiv 2022|[Code](https://github.com/jonkahana/CLIPPR)|
|[SgVA-CLIP: Semantic-guided Visual Adapting of Vision-Language Models for Few-shot Image Classification](https://arxiv.org/abs/2211.16191)|arXiv 2022|-|
|[SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More](https://arxiv.org/abs/2304.09148)|arXiv 2023|[Code](http://tianrun-chen.github.io/SAM-Adaptor/)|
|[Segment Anything in High Quality](https://arxiv.org/abs/2306.01567)|arXiv 2023|[Code](https://github.com/SysCV/SAM-HQ)|
|[HGCLIP: Exploring Vision-Language Models with Graph Representations for Hierarchical Understanding](https://arxiv.org/abs/2311.14064)|arXiv 2023|[Code](https://github.com/richard-peng-xia/HGCLIP)|
|[CLAP: Contrastive Learning with Augmented Prompts for Robustness on Pretrained Vision-Language Models](https://arxiv.org/abs/2311.16445)|arXiv 2023|-|

### Transfer with Other Methods

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[VT-Clip: Enhancing Vision-Language Models with Visual-guided Texts](https://arxiv.org/abs/2112.02399)|arXiv 2021|-|
|[Wise-FT: Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)|CVPR 2022|[Code](https://github.com/mlfoundations/wise-ft)|
|[MaskCLIP: Extract Free Dense Labels from CLIP](https://arxiv.org/abs/2112.01071)|ECCV 2022|[Code](https://github.com/chongzhou96/MaskCLIP)|
|[MUST: Masked Unsupervised Self-training for Label-free Image Classification](https://arxiv.org/abs/2206.02967)|ICLR 2023| [Code](https://github.com/salesforce/MUST)|
|[CALIP: Zero-Shot Enhancement of CLIP with Parameter-free Attention](https://arxiv.org/abs/2209.14169)|AAAI 2023|[Code](https://github.com/ziyuguo99/calip)|
|[Semantic Prompt for Few-Shot Image Recognition](https://arxiv.org/abs/2303.14123v1)|CVPR 2023|-|
|[Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners](https://arxiv.org/abs/2303.02151)|CVPR 2023|[Code](https://github.com/ZrrSkywalker/CaFo)|
|[Task Residual for Tuning Vision-Language Models](https://arxiv.org/abs/2211.10277)|CVPR 2023|[Code](https://github.com/geekyutao/TaskRes)|
|[Deeply Coupled Cross-Modal Prompt Learning](https://arxiv.org/abs/2305.17903)|ACL 2023|[Code](https://github.com/GingL/CMPA)|
|[Prompt Ensemble Self-training for Open-Vocabulary Domain Adaptation](https://arxiv.org/abs/2306.16658)|arXiv 2023|-|
|[Personalize Segment Anything Model with One Shot](https://arxiv.org/abs/2305.03048)|arXiv 2023|[Code](https://github.com/ZrrSkywalker/Personalize-SAM)|
|[Chils: Zero-shot image classification with hierarchical label sets](https://proceedings.mlr.press/v202/novack23a/novack23a.pdf)|ICML 2023|[Code](https://github.com/acmi-lab/CHILS)|
|[Improving Zero-shot Generalization and Robustness of Multi-modal Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Ge_Improving_Zero-Shot_Generalization_and_Robustness_of_Multi-Modal_Models_CVPR_2023_paper.pdf)|CVPR 2023|[Code](https://github.com/gyhandy/Hierarchy-CLIP)|
|[Exploiting Category Names for Few-Shot Classification with Vision-Language Models](https://openreview.net/pdf?id=w25Q9Ttjrs)|ICLR W 2023|-|
|[Beyond Sole Strength: Customized Ensembles for Generalized Vision-Language Models](https://arxiv.org/abs/2311.17091)|arXiv 2023|[Code](https://github.com/zhiheLu/Ensemble_VLM)|


## Vision-Language Model Knowledge Distillation Methods

### Knowledge Distillation for Object Detection
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[ViLD: Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921)|ICLR 2022|[Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)|
|[DetPro: Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model](https://arxiv.org/abs/2203.14940)|CVPR 2022|[Code](https://github.com/dyabel/detpro)|
|[XPM: Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling](https://arxiv.org/abs/2111.12698)|CVPR 2022|[Code](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling)|
|[Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection](https://arxiv.org/abs/2207.03482)|NeurIPS 2022|[Code](https://github.com/hanoonaR/object-centric-ovd)|
|[PromptDet: Towards Open-vocabulary Detection using Uncurated Images](https://arxiv.org/abs/2203.16513)|ECCV 2022|[Code](https://github.com/fcjian/PromptDet)|
|[PB-OVD: Open Vocabulary Object Detection with Pseudo Bounding-Box Labels](https://arxiv.org/abs/2111.09452)|ECCV 2022|[Code](https://github.com/salesforce/PB-OVD)|
|[OV-DETR: Open-Vocabulary DETR with Conditional Matching](https://arxiv.org/abs/2203.11876)|ECCV 2022|[Code](https://github.com/yuhangzang/OV-DETR)|
|[Detic: Detecting Twenty-thousand Classes using Image-level Supervision](https://arxiv.org/abs/2201.02605)|ECCV 2022|[Code](https://github.com/facebookresearch/Detic)|
|[OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230)|ECCV 2022|[Code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)|
|[VL-PLM: Exploiting Unlabeled Data with Vision and Language Models for Object Detection](https://arxiv.org/abs/2207.08954)|ECCV 2022|[Code](https://github.com/xiaofeng94/VL-PLM)|
|[ZSD-YOLO: Zero-shot Object Detection Through Vision-Language Embedding Alignment](https://arxiv.org/abs/2109.12066)|arXiv 2022|[Code](https://github.com/Johnathan-Xie/ZSD-YOLO)|
|[HierKD: Open-Vocabulary One-Stage Detection with Hierarchical Visual-Language Knowledge Distillation](https://arxiv.org/abs/2203.10593)|arXiv 2022|[Code](https://github.com/mengqiDyangge/HierKD)|
|[VLDet: Learning Object-Language Alignments for Open-Vocabulary Object Detection](https://arxiv.org/abs/2211.14843)|ICLR 2023|[Code](https://github.com/clin1223/VLDet)|
|[F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models](https://arxiv.org/abs/2209.15639)|ICLR 2023|[Code](https://github.com/google-research/google-research/tree/master/fvlm)|
|[CondHead: Learning to Detect and Segment for Open Vocabulary Object Detection](https://arxiv.org/abs/2212.12130)|CVPR 2023|-|
|[Aligning Bag of Regions for Open-Vocabulary Object Detection](https://arxiv.org/abs/2302.13996)|CVPR 2023|[Code](https://github.com/wusize/ovdet)|
|[Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2305.07011v1)|CVPR 2023|[Code](https://github.com/mcahny/rovit)|
|[Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection](https://arxiv.org/abs/2303.05892)|CVPR 2023|[Code](https://github.com/LutingWang/OADP)|
|[CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching](https://arxiv.org/abs/2303.13076v1)|CVPR 2023|[Code](https://github.com/tgxs002/CORA)|
|[DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment](https://arxiv.org/abs/2304.04514v1)|CVPR 2023|-|
|[Detecting Everything in the Open World: Towards Universal Object Detection](https://arxiv.org/abs/2303.11749)|CVPR 2023|[Code](https://github.com/zhenyuw16/UniDetector)|
|[CapDet: Unifying Dense Captioning and Open-World Detection Pretraining](https://arxiv.org/abs/2303.02489)|CVPR 2023|-|
|[Contextual Object Detection with Multimodal Large Language Models](https://arxiv.org/abs/2305.18279)|arXiv 2023|[Code](https://github.com/yuhangzang/ContextDET)|
|[Building One-class Detector for Anything: Open-vocabulary Zero-shot OOD Detection Using Text-image Models](https://arxiv.org/abs/2305.17207)|arXiv 2023|[Code](https://github.com/gyhandy/One-Class-Anything)|


### Knowledge Distillation for Semantic Segmentation

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[SSIW: Semantic Segmentation In-the-Wild Without Seeing Any Segmentation Examples](https://arxiv.org/abs/2112.03185)|arXiv 2021|-|
|[ReCo: Retrieve and Co-segment for Zero-shot Transfer](https://arxiv.org/abs/2206.07045)|NeurIPS 2022|[Code](https://github.com/NoelShin/reco)|
|[CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2203.02668)|CVPR 2022|[Code](https://github.com/CVI-SZU/CLIMS)|
|[CLIPSeg: Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003)|CVPR 2022|[Code](https://github.com/timojl/clipseg)|
|[ZegFormer: Decoupling Zero-Shot Semantic Segmentation](https://arxiv.org/abs/2112.07910)|CVPR 2022|[Code](https://github.com/dingjiansw101/ZegFormer)|
|[LSeg: Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546)|ICLR 2022|[Code](https://github.com/isl-org/lang-seg)|
|[ZSSeg: A Simple Baseline for Open-Vocabulary Semantic Segmentation with Pre-trained Vision-language Model](https://arxiv.org/abs/2112.14757)|ECCV 2022|[Code](https://github.com/MendelXu/zsseg.baseline)|
|[OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels](https://arxiv.org/abs/2112.12143)|ECCV 2022|[Code](https://github.com/tensorflow/tpu/tree/641c1ac6e26ed788327b973582cbfa297d7d31e7/models/official/detection/projects/openseg)|
|[Fusioner: Open-vocabulary Semantic Segmentation with Frozen Vision-Language Models](https://arxiv.org/abs/2210.15138)|BMVC 2022|[Code](https://github.com/chaofanma/Fusioner)|
|[OVSeg: Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP](https://arxiv.org/abs/2210.04150)|CVPR 2023|[Code](https://github.com/facebookresearch/ov-seg)|
|[ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation](https://arxiv.org/abs/2212.03588)|CVPR 2023|[Code](https://github.com/ZiqinZhou66/ZegCLIP)|
|[CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2212.09506)|CVPR 2023|[Code](https://github.com/linyq2117/CLIP-ES)|
|[FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation](https://arxiv.org/abs/2303.17225v1)|CVPR 2023|[Code](https://freeseg.github.io/)|
|[Mask-free OVIS: Open-Vocabulary Instance Segmentation without Manual Mask Annotations](https://arxiv.org/abs/2303.16891v1)|CVPR 2023|[Code](https://vibashan.github.io/ovis-web/)|
|[Exploring Open-Vocabulary Semantic Segmentation without Human Labels](https://arxiv.org/abs/2306.00450)|arXiv 2023|-|
|[OpenVIS: Open-vocabulary Video Instance Segmentation](https://arxiv.org/abs/2305.16835)|arXiv 2023|-|
|[Segment Anything is A Good Pseudo-label Generator for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2305.01275)|arXiv 2023|-|
|[Segment Anything Model (SAM) Enhanced Pseudo Labels for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2305.05803)|arXiv 2023|[Code](https://github.com/cskyl/SAM_WSSS)|
|[Plug-and-Play, Dense-Label-Free Extraction of Open-Vocabulary Semantic Segmentation from Vision-Language Models](https://arxiv.org/abs/2311.17095)|arXiv 2023|-|








