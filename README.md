# MeetAll · MeetBench · MeetMaster
![image](https://github.com/user-attachments/assets/ecd96592-b8b3-4705-8013-c8c687debc32)

> Multimodal & Multilingual Meeting Agent Suite  
> **ACM Multimedia 2025 – Dataset Track (under review)**

This repository hosts the complete, open–source implementation that accompanies our paper

> **"MeetBench: A Multimodal, Multilingual Meeting-Agent Dataset and Benchmark"**
![image](https://github.com/user-attachments/assets/f9c43081-7e9d-4121-abc9-447fdf42ee8b)

It contains three tightly–coupled components:

| Component | What it is | Location |
|-----------|------------|----------|
| **MeetAll** | 231 real-world meetings (≈ 140 h) with aligned transcripts, audio recordings, and 1 180 human-verified agent QA turns | `huggingface` |
| **MeetBench** | A multi-criteria benchmark (CompassJudger + Prometheus) for evaluating meeting assistants across factuality, user-need satisfaction, conciseness, structure and completeness | `./MeetBench_Benchmark/` |
| **MeetMaster** | A dual-process baseline agent (⚡ Talker + 🧠 Planner) that delivers both low-latency and strong reasoning | `./MeetMaster/` |

![image](https://github.com/user-attachments/assets/3984b31d-9738-40cb-8db1-5b13416665be)


All data, code, and pre-trained checkpoints will be released upon acceptance. Please ⭐ **Star** and **Watch** this repo to get notified!

---

## Table of Contents
1. [Features](#features)
2. [Directory Layout](#directory-layout)
3. [Quick Start](#quick-start)
4. [Reproducing the Paper](#reproducing-the-paper)
5. [Dataset & License](#dataset--license)
6. [Citation](#citation)
7. [Contributing](#contributing)
8. [Contact](#contact)
9. [Acknowledgements](#acknowledgements)

---

## Features
* **Rich data** – 200 h bilingual (English & Mandarin) audio with high-quality transcripts.
* **Diverse QA** – 2000 injected agent interactions covering 108 complexity cells across four cognitive axes (Cognitive Load, Context Dependency, Domain Knowledge, Task Effort).
* **Voice cloning** – Natural agent utterances generated with F5-TTS to ensure realistic meeting flows.
* **MeetBench** – First specialised benchmark for meeting assistants; integrates CompassJudger & Prometheus with meeting-specific prompts and scoring rubrics.
* **MeetMaster** – Dual-process architecture inspired by human fast–slow thinking: a light-weight **Planner** for routine queries and a reasoning-heavy **Talker** for easy ones.

---

## Directory Layout
```
MeetBench/

├── MeetBench_Benchmark/             # MeetBench evaluation framework
├── baseline/
│   └── MeetMaster/        # Talker & Planner implementation
├── experiment_result/     # Logs & metrics used in the paper
├── requirements.txt       # Python dependencies
└── meeting_simulator/     # End-to-end real-time meeting simulator

```

---

## Quick Start

### 1. Create the environment
```bash
# clone the repo
$ git clone https://github.com/MeetBench/MeetBench.git
$ cd MeetBench

# install dependencies
$ conda env create meetbench   # Python ≥3.10 / CUDA ≥11.7
$ conda activate meetbench
$ pip install -r requirements.txt
```



### 2. Download the MeetAll dataset
The full dataset is hosted on HuggingFace 

> Tip: Each shard ships with a SHA-256 checksum for integrity verification.

### 3. Evaluate an agent with MeetBench
```bash
$ python MeetBench_Benchmark/compare_results_compassJudger.py \
        --data_root ./MeetAll \
        --model meetmaster \
        --save_dir ./experiment_result
```

```bash
$ python MeetBench_Benchmark/compare_results_prometheus.py \
        --data_root ./MeetAll \
        --model meetmaster \
        --save_dir ./experiment_result


### 4. Live demo of MeetMaster
```bash
$ python MeetMaster/scripts/test_agent_audio.py.py 
# Type your question in the terminal and watch Talker & Planner respond in real-time
```
---

## Reproducing the Paper
The table below shows the main results reported in the paper.

Table 5: MeetBench Scores on MeetAll

| Model                      | Factual  | User Needs | Conciseness | Structure | Completeness | Final Score |
| -------------------------- | -------- | ---------- | ----------- | --------- | ------------ | ----------- |
| LLAMA-7B                   | 3.59     | 3.31       | 4.01        | 3.67      | 3.05         | 3.30        |
| LLAMA-13B                  | 5.58     | 5.07       | 6.14        | 6.08      | 4.77         | 5.13        |
| Qwen2.5-7B-Instruct        | 7.31     | 6.18       | 7.06        | 6.89      | 5.56         | 6.29        |
| chatGLM3-6B                | 6.01     | 5.29       | 6.33        | 6.17      | 4.91         | 5.44        |
| deepseek-r1-7B             | 7.32     | 6.43       | 7.74        | 7.21      | 5.91         | 6.50        |
| Qwen-Agent(Qwen2.5-7B api) | 7.44     | 6.53       | 7.72        | 7.24      | 6.21         | 6.56        |
| Phi-1                      | 5.38     | 5.27       | 6.12        | 5.13      | 6.17         | 4.27        |
| Phi-1.5                    | 5.98     | 5.63       | 6.17        | 5.68      | 6.34         | 5.67        |
| **MeetMaster**             | **7.50** | **6.57**   | **7.76**    | **7.33**  | **6.36**     | **6.59**    |


Table 6: Ablation Study Results on Five Dimensions of MeetBench

| Model        | Factual | User Needs | Conciseness | Structure | Completeness | Final Score |
| ------------ | ------- | ---------- | ----------- | --------- | ------------ | ----------- |
| Only Talker  | 5.96    | 5.25       | 6.27        | 6.03      | 5.00         | 5.38        |
| Only Planner | 7.99    | 6.99       | 8.32        | 7.76      | 6.39         | 7.05        |
| MeetMaster   | 7.50    | 6.57       | 6.76        | 7.33      | 6.36         | 6.59        |

Table 7: Overall Scores on MeetALL dataset
| Model                      | Prometheus Score |
| -------------------------- | ---------------- |
| **MeetMaster**             | **3.50**         |
| LLAMA-7B                   | 2.06             |
| LLAMA-13B                  | 3.35             |
| Qwen2-Audio                | 2.87             |
| Qwen2.5-7B-Instruct        | 3.43             |
| chatGLM3-6B                | 2.92             |
| deepseek-r1-7B             | 3.44             |
| Qwen-Agent(Qwen2.5-7B api) | 3.47             |
| Phi-1                      | 2.53             |
| Phi-1.5                    | 3.23             |

Table 8: Latency Measurements for MeetMaster
| Component                     | Latency (ms) |
| ----------------------------- | ------------ |
| STT Module (per token)        | 53           |
| Talker Latency (First Token)  | 210          |
| Talker Latency (Each Token)   | 31           |
| Planner Latency (First Token) | 520          |
| Planner Latency (Each Token)  | 310          |

Table 9: Ablation Study Results on Overall Score
| Model      | Prometheus Score |
| ---------- | ---------------- |
| MeetMaster | 3.50             |
| Talker     | 2.87             |
| Planner    | 3.69             |


---

## Dataset & License
* **Code** – Apache 2.0
* **Dataset (MeetAll)** – CC BY-NC 4.0 (non-commercial research only)

Please read `LICENSE` and `DATA_LICENSE` before use.

---

## Citation
If you find this work useful, please cite us:
```bibtex
@misc{meetall2025,
  title        = {MeetAll & MeetBench: A Multimodal, Multilingual Meeting-Agent Dataset and Benchmark},
  author       = {Your Name and Others},
  year         = {2025},
  note         = {Under review, ACM Multimedia Dataset Track},
  url          = {https://github.com/MeetBench/MeetBench}
}
```

---

## Contributing
We welcome contributions of any kind—bug fixes, new features, benchmarks, or documentation. Please read `CONTRIBUTING.md` and open a pull request.

---

## Contact
Questions? Feel free to open an issue or email us at **huyuelin@126.com**.

---

## Acknowledgements
This project heavily builds upon the open-source work of [F5-TTS](https://github.com/SWivid/F5-TTS), [CompassJudger](https://github.com/open-compass/CompassJudger), [Prometheus](https://github.com/prometheus-eval/prometheus-eval)and the broader research community. We thank all contributors! 
