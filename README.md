# [MGFN:Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection](https://arxiv.org/abs/2211.15098)
## This the official repo of paper accepted in AAAI 2023 Oral
### Pretrained models can be downloaded in here:[Onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EgbkWG-7TbFOnm28TLcyFaABHnniV6rcp_gzGm6OOVDWOQ?e=LrBlD5)
#### Prepare the environment: 
        $pip install -r requirement.txt
#### gt generate reference [github](https://github.com/ktr-hubrt/WSAL/blob/master/Test.py)
#### Test: Download the pretrained model and run $test.py
#### Dataset Prepare: [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE). Rename the data path in ucf-i3d.list and ucf-i3d-test.list based on your data path.
#### Train: Modify the option.py and run main.py
#### 32 segmented UCF dataset: [UCF-crime_ten-crop_I3D_32_seg](https://drive.google.com/drive/folders/1TfqCWvG3N2fqmiPIRuEkl_s2mNmbkRwN?usp=sharing). Rename the data path in ucf-i3d.list and ucf-i3d-test.list based on your data path.
#### Train: Use preprocessed flag if using the 32 seg data


## Citation
### If you find this repo useful for your research, please consider citing our paper:
      @misc{chen2022mgfn,
      title={MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection}, 
      author={Yingxian Chen and Zhengzhe Liu and Baoheng Zhang and Wilton Fok and Xiaojuan Qi and Yik-Chung Wu},
      year={2022},
      eprint={2211.15098},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
