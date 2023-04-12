# [MGFN:Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection](https://arxiv.org/abs/2211.15098)
## This the official repo of paper accepted in AAAI 2023 Oral
### Pretrained models and ground truth documents can be downloaded in here:[Onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EgbkWG-7TbFOnm28TLcyFaABHnniV6rcp_gzGm6OOVDWOQ?e=LrBlD5)
#### Prepare the environment: 
        $pip install -r requirement.txt
#### Test: Download the pretrained model and run $test.py
#### Dataset Prepare: [UCF-crime ten-crop I3D](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/cyxcarol_connect_hku_hk/EpNI-JSruH1Ep1su07pVLgIBnjDcBGd7Mexb1ERUVShdNg?e=VMRjhE). Rename the data path in ucf-i3d.list and ucf-i3d-test.list based on your data path.
#### Train: Modify the option.py and run main.py 

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
