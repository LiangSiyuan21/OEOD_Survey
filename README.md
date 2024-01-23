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

## Out of Domain Benchmark

### Data Manipulation

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




### Feature Learning
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Mega-cda: Memory guided attention for category-aware unsupervised domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2021/html/VS_MeGA-CDA_Memory_Guided_Attention_for_Category-Aware_Unsupervised_Domain_Adaptive_Object_CVPR_2021_paper.html)|CVPR 2021|-|
|[Domain adaptive faster R-CNN for object detection in the wild](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.html)|CVPR 2018|[Code](https://github.com/yuhuayc/da-faster-rcnn)|
|[Strong-weak distribution alignment for adaptive object detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.html)|CVPR 2019|[Code](https://github.com/VisionLearningGroup/DA_Detection)|
|[Adapting object detectors via selective cross-domain alignment](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Adapting_Object_Detectors_via_Selective_Cross-Domain_Alignment_CVPR_2019_paper.html)|CVPR 2019|[Code](https://github.com/xinge008/SCDA)|
|[Exploring categorical regularization for domain adaptive object detection](http://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Exploring_Categorical_Regularization_for_Domain_Adaptive_Object_Detection_CVPR_2020_paper.html)|CVPR 2020|[Code](https://github.com/megvii-research/CR-DA-DET)|
|[Spatial attention pyramid network for unsupervised domain adaptation](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_29)|ECCV 2020|[Code](https://github.com/Shuntw6096/sap-da-detectron2)|
|[Multi-adversarial faster-rcnn for unrestricted object detection](http://openaccess.thecvf.com/content_ICCV_2019/html/He_Multi-Adversarial_Faster-RCNN_for_Unrestricted_Object_Detection_ICCV_2019_paper.html)|ICCV 2019|[Code](https://github.com/He-Zhenwei/MAF)|
|[Domain-adaptive object detection via uncertainty- aware distribution alignment](https://dl.acm.org/doi/abs/10.1145/3394171.3413553)|MM 2020|[Code](https://github.com/basiclab/DA-OD-MEAA-PyTorch)|
|[Deeply aligned adaptation for cross-domain object detection](https://arxiv.org/abs/2004.02093)|arXiv 2020|-|
|[Domain adaptive object detection via asymmetric tri-way faster-rcnn](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_19)|ECCV 2020|-|
|[Adaptive object detection with dual multi-label prediction](https://link.springer.com/chapter/10.1007/978-3-030-58604-1_4)|ECCV 2020|-|
|[Seeking similarities over differences:Similarity-based domain alignment for adaptive object detection](http://openaccess.thecvf.com/content/ICCV2021/html/Rezaeianaran_Seeking_Similarities_Over_Differences_Similarity-Based_Domain_Alignment_for_Adaptive_Object_ICCV_2021_paper.html)|ICCV 2021|[Code](https://github.com/frezaeix/VISGA_Public)|
|[Decoupled adaptation for cross-domain object detection](https://arxiv.org/abs/2110.02578)|ICLR 2022|[Code](https://github.com/thuml/Decoupled-Adaptation-for-Cross-Domain-Object-Detection)|
|[Exploring sequence feature alignment for domain adaptive detection transformers](https://dl.acm.org/doi/abs/10.1145/3474085.3475317)|MM 2021|[Code](https://github.com/encounter1997/SFA)|
|[Informative consistent correspondence mining for cross-domain weakly supervised object detection](http://openaccess.thecvf.com/content/CVPR2021/html/Hou_Informative_and_Consistent_Correspondence_Mining_for_Cross-Domain_Weakly_Supervised_Object_CVPR_2021_paper.html)|CVPR 2021|-|
|[Task-specific inconsistency alignment for domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Task-Specific_Inconsistency_Alignment_for_Domain_Adaptive_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/MCG-NJU/TIA)|
|[Cross-domain detection via graph-induced prototype alignment](http://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Cross-Domain_Detection_via_Graph-Induced_Prototype_Alignment_CVPR_2020_paper.html)|CVPR 2020|[Code](https://github.com/ChrisAllenMing/GPA-detection)|
|[Cross-domain object detection through coarse-to-fine feature adaptation](http://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Cross-domain_Object_Detection_through_Coarse-to-Fine_Feature_Adaptation_CVPR_2020_paper.html)|CVPR 2020|-|
|[RPN prototype alignment for domain adaptive object detector](http://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RPN_Prototype_Alignment_for_Domain_Adaptive_Object_Detector_CVPR_2021_paper.html)|CVPR 2021|-|
|[Adapting object detectors with conditional domain normalization](https://link.springer.com/chapter/10.1007/978-3-030-58621-8_24)|ECCV 2020|[Code](https://github.com/zhoushiqi47/CDN)|
|[Vector-decomposed disentanglement for domain-invariant object detection](http://openaccess.thecvf.com/content/ICCV2021/html/Wu_Vector-Decomposed_Disentanglement_for_Domain-Invariant_Object_Detection_ICCV_2021_paper.html)|ICCV 2021|[Code](https://github.com/AmingWu/VDD-DAOD)|
|[Instance-invariant domain adaptive object detection via progressive disentanglement](https://ieeexplore.ieee.org/abstract/document/9362301/)|TPAMI 2022|[Code](https://github.com/AmingWu/IIOD)|
|[Single-domain generalized object detection in urban scene via cyclic-disentangled self-distillation](http://openaccess.thecvf.com/content/CVPR2022/html/Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation_CVPR_2022_paper.html)|CVPR 2022|-|

### Optimization Strategy
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Automatic adaptation of object detectors to new domains using self-training](http://openaccess.thecvf.com/content_CVPR_2019/html/RoyChowdhury_Automatic_Adaptation_of_Object_Detectors_to_New_Domains_Using_Self-Training_CVPR_2019_paper.html)|CVPR 2019|-|
|[A robust learning approach to domain adaptive object detection](http://openaccess.thecvf.com/content_ICCV_2019/html/Khodabandeh_A_Robust_Learning_Approach_to_Domain_Adaptive_Object_Detection_ICCV_2019_paper.html)|ICCV 2019|[Code](https://github.com/b02202050/robust_domain_adaptation)|
|[Self-training adversarial background regularization for unsupervised domain adaptive one-stage object detection](http://openaccess.thecvf.com/content_ICCV_2019/html/Kim_Self-Training_and_Adversarial_Background_Regularization_for_Unsupervised_Domain_Adaptive_One-Stage_ICCV_2019_paper.html)|ICCV 2019|-|
|[A free lunch for unsupervised domain adaptive object detection without source data](https://ojs.aaai.org/index.php/AAAI/article/view/17029)|AAAI 2021|-|
|[Category dictionary guided unsupervised domain adaptation for object detection](https://ojs.aaai.org/index.php/AAAI/article/view/16290)|AAAI 2021|[Code](https://github.com/merlinarer/CDG)|
|[SSAL: synergizing between self-training adversarial learning for domain adaptive object detection](https://proceedings.neurips.cc/paper/2021/hash/c0cccc24dd23ded67404f5e511c342b0-Abstract.html)|NIPS 2021|-|
|[Multi-view domain adaptive object detection on camera networks](https://yshu.org/paper/aaai23mvdaod.pdf)|AAAI 2023|-|
|[Exploring object relation in mean teacher for cross domain detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Cai_Exploring_Object_Relation_in_Mean_Teacher_for_Cross-Domain_Detection_CVPR_2019_paper.html)|CVPR 2019|[Code](https://github.com/caiqi/mean-teacher-cross-domain-detection)|
|[Unbiased teacher for semi-supervised object detection](https://arxiv.org/abs/2102.09480)|ICLR 2021|[Code](https://github.com/facebookresearch/unbiased-teacher)|
|[Unbiased mean teacher for cross-domain object detection](http://openaccess.thecvf.com/content/CVPR2021/html/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.html)|CVPR 2021|[Code](https://github.com/kinredon/umt)|
|[Cross domain object detection by target-perceived dual branch distillation](http://openaccess.thecvf.com/content/CVPR2022/html/He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation_CVPR_2022_paper.html)|CVPR 2022|-|
|[Cross-domain adaptive teacher for object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/facebookresearch/adaptive_teacher)|
|[Target-relevant knowledge preservation for multi-source domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|-|
|[Multi-source domain adaptation for object detection](http://openaccess.thecvf.com/content/ICCV2021/html/Yao_Multi-Source_Domain_Adaptation_for_Object_Detection_ICCV_2021_paper.html)|ICCV 2021|[Code](https://github.com/jh-Han777/Multi_Source_Domain_Adaptation_for_Object_Detection)|
|[2pcnet: Two-phase consistency training for day-to-night unsupervised domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2023/html/Kennerley_2PCNet_Two-Phase_Consistency_Training_for_Day-to-Night_Unsupervised_Domain_Adaptive_Object_CVPR_2023_paper.html)|CVPR 2023|-|

## Out of Category Benchmark

### Discriminant
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Expanding low-density latent regions for open-set object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Han_Expanding_Low-Density_Latent_Regions_for_Open-Set_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/csuhan/opendet2)|
|[Dropout sampling for robust object detection in open-set conditions](https://ieeexplore.ieee.org/abstract/document/8460700/)|ICRA 2018|-|
|[Evaluating merging strategies for sampling-based uncertainty techniques in object detection](https://ieeexplore.ieee.org/abstract/document/8793821/)|ICRA 2019|-|
|[Vos: Learning what you don’t know by virtual outlier synthesis](https://arxiv.org/abs/2202.01197)|ICLR 2022|[Code](https://github.com/deeplearning-wisc/vos)|
|[Towards open world object detection](http://openaccess.thecvf.com/content/CVPR2021/html/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.html)|CVPR 2021|[Code](https://github.com/AIX-Coast-Defense-PIL/Towards-Open-World-Object-Detection)|
|[Unknown-aware object detection: Learning what you don’t know from videos in the wild](http://openaccess.thecvf.com/content/CVPR2022/html/Du_Unknown-Aware_Object_Detection_Learning_What_You_Dont_Know_From_Videos_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/deeplearning-wisc/stud)|
|[Open-set semi-supervised object detection](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_9)|ECCV 2022|-|
|[Proposalclip: unsupervised open-category object proposal generation via exploiting clip cues](http://openaccess.thecvf.com/content/CVPR2022/html/Shi_ProposalCLIP_Unsupervised_Open-Category_Object_Proposal_Generation_via_Exploiting_CLIP_Cues_CVPR_2022_paper.html)|CVPR 2022|-|
|[Uc-owod: Unknown-classified open world object detection](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_12)|ECCV 2022|-|
|[Hsic-based moving weight averaging for few-shot open-set object detection](https://dl.acm.org/doi/abs/10.1145/3581783.3611850)|MM 2023|-|

### Side Information
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Zero shot detection](https://ieeexplore.ieee.org/abstract/document/8642945/)|TCSVT 2019|[Code](https://github.com/nasir6/zero_shot_detection)|
|[Zero-shot object detection with attributes-based category similarity](https://ieeexplore.ieee.org/abstract/document/9043901/)|TCSVT 2020|-|
|[Zero-shot object detection](http://openaccess.thecvf.com/content_ECCV_2018/html/Ankan_Bansal_Zero-Shot_Object_Detection_ECCV_2018_paper.html)|ECCV 2018|-|
|[Zero-shot object detection with textual descriptions](https://ojs.aaai.org/index.php/AAAI/article/view/4891)|AAAI 2019|-|
|[Semantics-preserving graph propagation for zero-shot object detection](https://ieeexplore.ieee.org/abstract/document/9153181/)|TIP 2020|-|
|[Polarity loss: Improving visual_semantic alignment for zero-shot detection](https://ieeexplore.ieee.org/abstract/document/9812473/)|TNNLS 2022||
|[From node to graph: Joint reasoning on visual_semantic relational graph for zero-shot detection](http://openaccess.thecvf.com/content/WACV2022/html/Nie_From_Node_To_Graph_Joint_Reasoning_on_Visual_Semantic_Relational_Graph_WACV_2022_paper.html)|WACV 2022|[Code](https://github.com/witnessai/GRAN)|
|[Semantics-guided contrastive network for zero-shot object detection](https://ieeexplore.ieee.org/abstract/document/9669022/)|TPAMI 2022|-|
|[Robust region feature synthesizer for zero-shot object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Huang_Robust_Region_Feature_Synthesizer_for_Zero-Shot_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|-|
|[Generative multi-label zero-shot learning](https://ieeexplore.ieee.org/abstract/document/10184028/)|TPAMI 2023|[Code](https://github.com/akshitac8/Generative_MLZSL)|
|[Zero-shot camouflaged object detection](https://ieeexplore.ieee.org/abstract/document/10234216/)|TIP 2023|-|

### Arbitrary Information
| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Open-vocabulary object detection using captions](https://openaccess.thecvf.com/content/CVPR2021/html/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.html?ref=https://githubhelp.com)|CVPR 2021|-|
|[Open-vocabulary object detection via vision and language knowledge distillation](https://arxiv.org/abs/2104.13921)|ICLR 2022|-|
|[Open-vocabulary one-stage detection with hierarchical visual-language knowledge distillation](http://openaccess.thecvf.com/content/CVPR2022/html/Ma_Open-Vocabulary_One-Stage_Detection_With_Hierarchical_Visual-Language_Knowledge_Distillation_CVPR_2022_paper.html)|CVPR 2022|-|
|[F-vlm: Open-vocabulary object detection upon frozen vision and language models](https://arxiv.org/abs/2209.15639)|ICLR 2023|-|
|[Open Vocabulary Object Detection with Pseudo Bounding-Box Labels](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_16)|ECCV 2022|[Code](https://github.com/salesforce/PB-OVD)|
|[PromptDet: Towards Open-vocabulary Detection using Uncurated Images](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_41)|ECCV 2022|[Code](https://github.com/fcjian/PromptDet)|
|[Grounded language-image pre-training](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html?ref=blog.roboflow.com)|CVPR 2022|[Code](https://github.com/microsoft/GLIP)|
|[Regionclip: Region-based language-image pretraining](http://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/microsoft/RegionCLIP)|
|[Detecting everything in the open world: Towards universal object detection](http://openaccess.thecvf.com/content/CVPR2023/html/Wang_Detecting_Everything_in_the_Open_World_Towards_Universal_Object_Detection_CVPR_2023_paper.html)|CVPR 2023|[Code](https://github.com/zhenyuw16/UniDetector)|
|[Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers](http://openaccess.thecvf.com/content/CVPR2023/html/Kim_Region-Aware_Pretraining_for_Open-Vocabulary_Object_Detection_With_Vision_Transformers_CVPR_2023_paper.html)|CVPR 2023|[Code](https://github.com/mcahny/rovit)|
|[DetCLIP: Dictionary-enriched visual-concept paralleled pre-training for open-world detection](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ba960559212691be13fa81d9e5e0047-Abstract-Conference.html)|NIPS 2022|-|
|[DetCLIPv2: Scalable open-vocabulary object detection pre-training via word-region alignment](http://openaccess.thecvf.com/content/CVPR2023/html/Yao_DetCLIPv2_Scalable_Open-Vocabulary_Object_Detection_Pre-Training_via_Word-Region_Alignment_CVPR_2023_paper.html)|CVPR 2023|-|

## Malicious Data Benchmark

### Adversarial Training

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Using feature alignment can improve clean average precision and adversarial robustness in object detection](https://ieeexplore.ieee.org/abstract/document/9506689/)|ICIP 2021|-|
|[On the importance of backbone to the adversarial robustness of object detectors](https://arxiv.org/abs/2305.17438)|arXiv 2023|-|
|[Adversarial attack and defense of yolo detectors in autonomous driving scenarios](https://ieeexplore.ieee.org/abstract/document/9827222/)|IV 2022|-|
|[Towards adversarially robust object detection](http://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Towards_Adversarially_Robust_Object_Detection_ICCV_2019_paper.html)|ICCV 2019|-|
|[Robust and accurate object detection via adversarial learning](http://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_and_Accurate_Object_Detection_via_Adversarial_Learning_CVPR_2021_paper.html)|CVPR 2021|-|
|[Robust and accurate object detection via self knowledge distillation](https://ieeexplore.ieee.org/abstract/document/9898031/)|ICIP 2022|-|
|[Adversarially-aware robust object detector](https://link.springer.com/chapter/10.1007/978-3-031-20077-9_18)|ECCV 2022|[Code](https://github.com/IrisRainbowNeko/RobustDet)|
|[Adversarial intensity awareness for robust object detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4279328)|SSRN 2022|-|
|[Towards efficient adversarial training on vision transformers](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_18)|ECCV 2022|-|
|[Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994)|arXiv 2020|-|
|[Class-aware robust adversarial training for object detection](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Class-Aware_Robust_Adversarial_Training_for_Object_Detection_CVPR_2021_paper.html?ref=https://githubhelp.com)|CVPR 2021|-|

### Model Robust Inference

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[A study of the effect of jpg compression on adversarial images](https://arxiv.org/abs/1608.00853)|arXiv 2016|-|
|[Adversarial pixel masking: A defense against physical attacks for pre-trained object detectors](https://dl.acm.org/doi/abs/10.1145/3474085.3475338)|MM 2021|-|
|[Local gradients smoothing: Defense against localized adversarial attacks](https://ieeexplore.ieee.org/abstract/document/8658401/)|WACV 2019|[Code](https://github.com/icyham/local_gradients_smoothing)|
|[Feature squeezing: Detecting adversarial examples in deep neural networks](https://arxiv.org/abs/1704.01155)|arXiv 2017|-|
|[Detection as regression: Certified object detection with median smoothing](https://proceedings.neurips.cc/paper/2020/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html)|NIPS 2020|-|
|[Detectorguard: Provably securing object detectors against localized patch hiding attacks](https://dl.acm.org/doi/abs/10.1145/3460120.3484757)|CCS 2021|[Code](https://github.com/inspire-group/DetectorGuard)|
|[Real-time robust video object detection system against physical-world adversarial attacks](https://ieeexplore.ieee.org/abstract/document/10220201/)|TCAD 2023|-|
|[Segment and complete: Defending object detectors against adversarial patch attacks with robust patch detection](http://openaccess.thecvf.com/content/CVPR2022/html/Liu_Segment_and_Complete_Defending_Object_Detectors_Against_Adversarial_Patch_Attacks_CVPR_2022_paper.html)|CVPR 2022|-|
|[Defending from physically-realizable adversarial attacks through internal over-activation analysis](https://ojs.aaai.org/index.php/AAAI/article/view/26758)|AAAI 2023|-|




## Incremental Data Benchmark

### Replay Based

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Rt-net: replay-and-transfer network for class incremental object detection](https://link.springer.com/article/10.1007/s10489-022-03509-0)|Applied Intelligence, 2023|-|
|[Continual detection transformer for incremental object detection](http://openaccess.thecvf.com/content/CVPR2023/html/Liu_Continual_Detection_Transformer_for_Incremental_Object_Detection_CVPR_2023_paper.html)|CVPR 2023|[Code](https://github.com/yaoyao-liu/CL-DETR)|
|[Towards open world object detection](http://openaccess.thecvf.com/content/CVPR2021/html/Joseph_Towards_Open_World_Object_Detection_CVPR_2021_paper.html)|CVPR 2021|[Code](https://github.com/AIX-Coast-Defense-PIL/Towards-Open-World-Object-Detection)|
|[Continual learning strategy in one-stage object detection framework based on experience replay for autonomous driving vehicle](https://www.mdpi.com/1424-8220/20/23/6777)|Sensors 2020|-|
|[Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection](http://openaccess.thecvf.com/content/ICCV2023/html/Liu_Augmented_Box_Replay_Overcoming_Foreground_Shift_for_Incremental_Object_Detection_ICCV_2023_paper.html)|ICCV 2023|-|
|[One-shot replay: boosting incremental object detection via retrospecting one object](https://ojs.aaai.org/index.php/AAAI/article/view/25417)|AAAI 2023|-|
|[Rodeo: Replay for online object detection](https://arxiv.org/abs/2008.06439)|arXiv 2020|[Code](https://github.com/manoja328/rodeo)|
|[Utilizing incremental branches on a one-stage object detection framework to avoid catastrophic forgetting](https://link.springer.com/article/10.1007/s00138-022-01284-z)|MVA 2022|-|
|[An end-to-end architecture for class-incremental object detection with knowledge distillation](https://ieeexplore.ieee.org/abstract/document/8784755/)|ICME 2019|-|
|[Incremental few-shot instance segmentation](http://openaccess.thecvf.com/content/CVPR2021/html/Ganea_Incremental_Few-Shot_Instance_Segmentation_CVPR_2021_paper.html)|CVPR 2021|[Code](https://github.com/danganea/iMTFA)|



### Model Based

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Incremental learning of single-stage detectors with mining memory neurons](https://ieeexplore.ieee.org/abstract/document/8780739/)|ICCC 2018|-|
|[Diode: dilatable incremental object detection](https://www.sciencedirect.com/science/article/pii/S0031320322007233)|PR 2023|-|
|[Lstd: A low-shot transfer detector for object detection](https://ojs.aaai.org/index.php/AAAI/article/view/11716)|AAAI 2018|-|
|[Few-shot object detection via feature reweighting](http://openaccess.thecvf.com/content_ICCV_2019/html/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.html)|ICCV 2019|[Code](https://github.com/bingykang/Fewshot_Detection)|
|[Meta-learning-based incremental few-shot object detection](https://ieeexplore.ieee.org/abstract/document/9452164/)|TCSVT 2021|[Code](https://github.com/Tongji-MIC-Lab/ML-iFSOD)|
|[Sylph: A hypernetwork framework for incremental few-shot object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Yin_Sylph_A_Hypernetwork_Framework_for_Incremental_Few-Shot_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|-|
|[Incremental few-shot object detection](http://openaccess.thecvf.com/content_CVPR_2020/html/Perez-Rua_Incremental_Few-Shot_Object_Detection_CVPR_2020_paper.html)|CVPR 2020|-|
|[Few-shot batch incremental road object detection via detector fusion](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/html/Tambwekar_Few-Shot_Batch_Incremental_Road_Object_Detection_via_Detector_Fusion_ICCVW_2021_paper.html)|ICCVW 2021|-|
|[Context-transformer: Tackling object confusion for few-shot detection](https://ojs.aaai.org/index.php/AAAI/article/view/6957)|AAAI 2020|[Code](https://github.com/Ze-Yang/Context-Transformer)|
|[Incremental learning of object detection with output merging of compact expert detectors](https://ieeexplore.ieee.org/abstract/document/9527693/)|ICoIAS 2021|-|
|[Continual object detection via prototypical task correlation guided gating mechanism](http://openaccess.thecvf.com/content/CVPR2022/html/Yang_Continual_Object_Detection_via_Prototypical_Task_Correlation_Guided_Gating_Mechanism_CVPR_2022_paper.html)|CVPR 2022|-|
|[Dlcft: Deep linear continual fine-tuning for general incremental learning](https://link.springer.com/chapter/10.1007/978-3-031-19827-4_30)|ECCV 2022|-|
|[Side-tuning: a baseline for network adaptation via additive side networks](https://link.springer.com/chapter/10.1007/978-3-030-58580-8_41)|ECCV 2020|-|
|[Any-shot object detection](https://openaccess.thecvf.com/content/ACCV2020/html/Rahman_Any-Shot_Object_Detection_ACCV_2020_paper.html)|ACCV 2020|-|
|[Incremental-detr: Incremental few-shot object detection via self-supervised learning](https://ojs.aaai.org/index.php/AAAI/article/view/25129)|AAAI 2023|-|
|[Ow-detr: Open-world detection transformer](http://openaccess.thecvf.com/content/CVPR2022/html/Gupta_OW-DETR_Open-World_Detection_Transformer_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/akshitac8/OW-DETR)|

### Regularization Based

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Exploring object relation in mean teacher for cross-domain detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Cai_Exploring_Object_Relation_in_Mean_Teacher_for_Cross-Domain_Detection_CVPR_2019_paper.html)|CVPR 2019|[Code](https://github.com/caiqi/mean-teacher-cross-domain-detection)|
|[Unbiased teacher for semi-supervised object detection](https://arxiv.org/abs/2102.09480)|ICLR 2021|[Code](https://github.com/facebookresearch/unbiased-teacher)|
|[Unbiased mean teacher for cross-domain object detection](http://openaccess.thecvf.com/content/CVPR2021/html/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.html)|CVPR 2021|[Code](https://github.com/kinredon/umt)|
|[Cross domain object detection by target-perceived dual branch distillation](http://openaccess.thecvf.com/content/CVPR2022/html/He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation_CVPR_2022_paper.html)|CVPR 2022|-|
|[Cross-domain adaptive teacher for object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Adaptive_Teacher_for_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/facebookresearch/adaptive_teacher)|
|[Target-relevant knowledge preservation for multi-source domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2022/html/Wu_Target-Relevant_Knowledge_Preservation_for_Multi-Source_Domain_Adaptive_Object_Detection_CVPR_2022_paper.html)|CVPR 2022|-|
|[Multi-source domain adaptation for object detection](http://openaccess.thecvf.com/content/ICCV2021/html/Yao_Multi-Source_Domain_Adaptation_for_Object_Detection_ICCV_2021_paper.html)|ICCV 2021|[Code](https://github.com/jh-Han777/Multi_Source_Domain_Adaptation_for_Object_Detection)|
|[2pcnet: Two-phase consistency training for day-to-night unsupervised domain adaptive object detection](http://openaccess.thecvf.com/content/CVPR2023/html/Kennerley_2PCNet_Two-Phase_Consistency_Training_for_Day-to-Night_Unsupervised_Domain_Adaptive_Object_CVPR_2023_paper.html)|CVPR 2023|-|
|[Balanced ranking and sorting for class incremental object detection](https://ieeexplore.ieee.org/abstract/document/9747449/)|ICASSP 2022|-|
|[Incdet: In defense of elastic weight consolidation for incremental object detection](https://ieeexplore.ieee.org/abstract/document/9127478/)|TNNLS 2020|-|
|[ifs-rcnn: An incremental few-shot instance segmenter](http://openaccess.thecvf.com/content/CVPR2022/html/Nguyen_iFS-RCNN_An_Incremental_Few-Shot_Instance_Segmenter_CVPR_2022_paper.html)|CVPR 2022|[Code](https://github.com/ducminhkhoi/iFS-RCNN)|

### Optimization Based

| Paper                                             |  Published in | Code/Project |                                  
|---------------------------------------------------|:-------------:|:------------:|
|[Towards generalized and incremental few-shot object detection](https://arxiv.org/abs/2109.11336)|arXiv 2021|-|
|[Incremental object detection via meta-learning](https://ieeexplore.ieee.org/abstract/document/9599446/)|TPAMI 2021|[Code](https://github.com/JosephKJ/iOD)|
|[Fast hierarchical learning for few-shot object detection](https://ieeexplore.ieee.org/abstract/document/9981327/)|IROS 2022|[Code](https://github.com/yihshe/fast-hierarchical-learning-for-fsod)|








