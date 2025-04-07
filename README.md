# Efficient Multi-task Deep Learning for Accurate Segmentation of Invasive and Core Tumor Regions in Spheroids




## Requirement
See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```


## Data Preparation
- Download the images from [dataset1]https://drive.google.com/file/d/1DolDF7l_ZpU1-RiluAa0Zq4aP9_SWk40/view?usp=sharing
- Download the images from [dataset2]https://drive.google.com/file/d/15PpnkILWMax7qyczGy7-bgWJukL5eTZX/view?usp=sharing

```bash
/data
    Tumor_dataset
        images
            train/
            val/
            test/
        Core
            train/
            val/
            test/
        Images
            train/
            val/
            test/
```
## Pipeline

<div align=center>
<img src='image\arch.png' width='600'>
</div>

## Train
```python
python3 main.py
```

## Test
```python
python3 val.py
```

## Inference

### Images
```python
python3 test_image.py
```





## Citation

If you find our paper and code useful for your research, please consider giving a star :star:   and citation :pencil: :

```BibTeX
@INPROCEEDINGS{10288646,
  author={},
  Journal={2025 The Visual Computer)}, 
  title={Efficient Multi-task Deep Learning for Accurate Segmentation of Invasive and Core Tumor Regions in Spheroids}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
```

<div align="center">
  <img src="twin.png" width="30%">
</div>
