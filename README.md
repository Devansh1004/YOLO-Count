<p align="center">
  <h1 align="center">YOLO-Count: Differentiable Object Counting for Text-to-Image Generation</h1>

  <p align="center">
    <a href="https://scholar.google.com/citations?user=SU6ooAQAAAAJ&hl=en" target="_blank"><strong>Guanning Zeng</strong></a>
    ·
    <a href="https://xzhang.dev/" target="_blank"><strong>Xiang Zhang</strong></a>
    ·
    <a href="https://zwcolin.github.io/" target="_blank"><strong>Zirui Wang</strong></a>
    ·
    <a href="https://xxuhaiyang.github.io/" target="_blank"><strong>Haiyang Xu</strong></a>
    ·
    <a href="https://zeyuan-chen.com/" target="_blank"><strong>Zeyuan Chen</strong></a>
    ·
    <a href="https://www.bingnanli.com/" target="_blank"><strong>Bingnan Li</strong></a>
    ·
    <a href="https://pages.ucsd.edu/~ztu/" target="_blank"><strong>Zhuowen Tu</strong></a>
  </p>

  <p align="center">
    <strong><i>ICCV 2025</i></strong>
  </p>
</p>

<h3 align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_YOLO-Count_Differentiable_Object_Counting_for_Text-to-Image_Generation_ICCV_2025_paper.pdf">Paper</a>
  |
  <a href="https://arxiv.org/abs/2508.00728">arXiv</a>
</h3>

<div align="center">
  <img src="figures/pipeline.webp" alt="Pipeline" width="100%">
</div>

---

This repository contains the official implementation of **YOLO-Count**, a fully differentiable and open-vocabulary object counting model. YOLO-Count is designed to provide accurate object count estimation and enable fine-grained quantity control for text-to-image (T2I) generation models.

---

## Environment Preparation

We recommend using Conda to set up the environment.

```bash
conda create -n yolocnt python=3.12
conda activate yolocnt
pip install -r requirements.txt
```

---

## Dataset Preparation

YOLO-Count is trained and evaluated on multiple object counting benchmarks. Please download and organize each dataset as follows.

### FSC147
- Download **FSC147** from  
  https://github.com/cvlab-stonybrook/LearningToCountEverything
- Place the following folders under:
  ```text
  data/FSC/
  ├── gt_density_map_adaptive_384_VarV2
  └── images_384_VarV2
  ```

### Open Images v7 (OImgv7)
Download Open Images v7 using:
```bash
python -m scripts.download_oimgv7
```

### Objects365 (Obj365)
Download the validation images with:
```bash
python -m scripts.download_o365
```

Then organize the data as:
```text
data/Obj365/objects365/val
```

### LVIS
- Download **LVIS**
- Place all files under:
  ```text
  data/LVIS/
  ```

---

## Pre-trained Weights

Pre-trained model weights are available at  
https://huggingface.co/zx1239856/yolo-count/tree/main

Please download the weights and place them in the `checkpoints/` directory.

---

## Evaluation

Evaluation can be performed using the `eval_*.py` scripts in the `scripts` folder.  
For example, to evaluate on FSC147:

```bash
python -m scripts.eval_fsc
```

### Results

The table below reports counting performance using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

| Dataset  | Split       | MAE     | RMSE     |
|----------|-------------|---------|----------|
| FSC      | Test        | 15.6745 | 96.3807  |
| FSC      | Validation  | 14.8297 | 59.6979  |
| LVIS     | Validation  | 1.5379  | 5.6076   |
| OImgv7   | Validation  | 3.7087  | 12.0285  |
| Obj365   | Validation  | 3.2749  | 9.2181   |

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@InProceedings{zeng2025yolocount,
    author    = {Zeng, Guanning and Zhang, Xiang and Wang, Zirui and Xu, Haiyang and Chen, Zeyuan and Li, Bingnan and Tu, Zhuowen},
    title     = {YOLO-Count: Differentiable Object Counting for Text-to-Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {16765--16775}
}
```

---

## License

This repository is released under the [CC-BY-SA 4.0](LICENSE) license.
