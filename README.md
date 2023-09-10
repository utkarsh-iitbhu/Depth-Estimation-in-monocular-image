# Summer Internship Report
## Data Scientist Internship at Publicis Sapient
### May 2023 - July 2023

**Table of Contents**
1. **Introduction**
   - Overview
   - Objectives

2. **Project Highlights**
   - Clustering for Facility Optimization
   - Mathematical Models with Pulp
   - Inventory and Route Optimization
   - Deployment on AWS Kubernetes Cluster

3. **Key Learnings and Exposure**
   - Operational Research
   - Business Insights
   - Mathematical Modeling with Pulp
   - Kubernetes Orchestration
   - Streamlit Application Development

4. **Conclusion**

---

### 1. Introduction

#### Overview

During my summer internship at Publicis Sapient from May 2023 to July 2023, I had the opportunity to work as a Data Scientist. It was an enriching experience where I contributed to solving complex business problems using data-driven approaches and mathematical models. This report summarizes my key contributions and learnings during this internship.

#### Objectives

The primary objectives of my internship were as follows:

- Develop clustering and mathematical models to optimize facility locations in warehouse management.
- Optimize the objective function to address inventory and route optimization challenges in the delivery chain.
- Deploy code efficiently using Docker and manage deployment and service pods on an AWS Kubernetes cluster.

### 2. Project Highlights

#### Clustering for Facility Optimization

One of the significant projects I worked on involved optimizing facility locations. By implementing clustering algorithms, we were able to identify the optimal locations for warehouses. This approach not only reduced operational costs but also improved delivery efficiency.

#### Mathematical Models with Pulp

I gained valuable experience in mathematical modeling using Pulp, a Python library for linear and mixed-integer programming. We developed and fine-tuned mathematical models that addressed complex optimization challenges within the company's supply chain.

#### Inventory and Route Optimization

The internship involved addressing inventory management and route optimization issues in the delivery chain. By applying mathematical modeling techniques, we optimized the allocation of inventory and designed efficient delivery routes. This had a direct impact on reducing costs and improving service quality.

#### Deployment on AWS Kubernetes Cluster

To ensure scalability and reliability, we deployed our optimization code using Docker containers on an AWS Kubernetes cluster. This allowed us to manage deployment and service pods effectively, ensuring that our solutions could handle real-world demands seamlessly.

### 3. Key Learnings and Exposure

#### Operational Research

I gained a deep understanding of operational research concepts, which played a pivotal role in solving complex logistics and supply chain problems. This knowledge allowed me to approach problems systematically and find optimal solutions.

#### Business Insights

Working closely with the team, I gained insights into the business domain, understanding the intricacies of supply chain management and warehouse operations. This exposure was invaluable in tailoring data-driven solutions to meet real-world business needs.

#### Mathematical Modeling with Pulp

My internship exposed me to mathematical modeling using Pulp. I learned how to formulate optimization problems, create objective functions, and find solutions using linear and mixed-integer programming. This skillset has broad applications in various industries.

#### Kubernetes Orchestration

Managing code deployment using Docker containers on an AWS Kubernetes cluster was an exciting learning experience. It provided insights into cloud-based infrastructure and orchestration, which is increasingly vital in today's technology landscape.

#### Streamlit Application Development

As part of a supplementary project, I developed interactive Streamlit applications to visualize and present our optimization solutions. This enhanced my data presentation and communication skills, making it easier to convey complex ideas to non-technical stakeholders.

### 4. Conclusion

My summer internship at Publicis Sapient as a Data Scientist was a rewarding journey that equipped me with valuable skills and knowledge. The exposure to real-world business challenges, coupled with the opportunity to develop and implement data-driven solutions, has been instrumental in my professional growth.

I am grateful for the guidance of my mentors and the support of the entire team at Publicis Sapient. This internship has reinforced my passion for data science and its transformative potential in solving complex problems. I look forward to applying these skills in future endeavors, contributing to data-driven innovations in the world of business and technology.




## Vision Transformers for Dense Prediction

### RESULTS:

![image](https://user-images.githubusercontent.com/84759422/210115514-980d22ed-1fb0-4411-b21b-bc9e2286edab.png)

![image](https://user-images.githubusercontent.com/84759422/210115479-36c9ed10-eb81-40df-a2a9-2ceee318e9ad.png)

### Depth estimation in monocular images


This repository contains code and models for our [paper](https://arxiv.org/abs/2103.13413):


### Setup 

1) Download the model weights and place them in the `weights` folder:


Monodepth:
- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt), [Mirror](https://drive.google.com/file/d/1vnuhoMc6caF-buQQ4hK0CeiMk9SjwB-G/view?usp=sharing)
2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.7, PyTorch 1.8.0, OpenCV 4.5.1, and timm 0.4.5

### Usage 

1) Place one or more input images in the folder `input`.

2) Run a monocular depth estimation model:

    ```shell
    python run_monodepth.py
    ```

3) The results are written to the folder `output_monodepth`.

Use the flag `-t` to switch between different models. Possible options are `dpt_hybrid` (default) and `dpt_large`.


**Additional models:**

- Monodepth finetuned on KITTI: [dpt_hybrid_kitti-cb926ef4.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_kitti-cb926ef4.pt) [Mirror](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view?usp=sharing)
- Monodepth finetuned on NYUv2: [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) [Mirror](https\://drive.google.com/file/d/1NjiFw1Z9lUAfTPZu4uQ9gourVwvmd58O/view?usp=sharing)

Run with 

```shell
python run_monodepth -t [dpt_hybrid_kitti|dpt_hybrid_nyu] 
```

### Evaluation

Hints on how to evaluate monodepth models can be found here: https://github.com/intel-isl/DPT/blob/main/EVALUATION.md


