## **TalkingStyle**

Official PyTorch implementation for the paper:

> **TalkingStyle: Personalized Speech-Driven Facial Animation with Style Preservation** [TVCG]
>
> Wenfeng Song, Xuan Wang, Shi Zheng, Shuai Li, Aimin Hao, Xia Hou

<p align="center">
<img src="fig/pipline.png" width="100%"/>
</p>


> This paper presents “TalkingStyle”, a novel method for generating personalized talking avatars while preserving speech content. Our approach uses a set of audio and animation samples from an individual to create new facial animations that closely resemble their specific talking style, synchronized with speech. 


## **Environmental Preparation**

- Ubuntu (Linux)
- Python 3.8+
- Pytorch 1.9.1+
- CUDA 13.1 (GPU with at least 11GB VRAM)
- ffmpeg
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

Other necessary packages:
```
pip install -r requirements.txt
```


## **Dataset Preparation**
### VOCASET
Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `vocaset/`. Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `vocaset/`. Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav`:
```
cd vocaset
python process_voca_data.py
```

### BIWI

Follow the [`BIWI/README.md`](BIWI/README.md) to preprocess BIWI dataset and put .npy/.wav files into `BIWI/vertices_npy` and `BIWI/wav`, and the `templates.pkl` into `BIWI/`.


## **Demo**
Download the pretrained models from [biwi.pth](https://drive.google.com/drive/folders/1IYVyYp35ueNbXZ3XFbugFMYhTw6PBVJ8?usp=drive_link), [vocaset.pth](https://drive.google.com/drive/folders/1IYVyYp35ueNbXZ3XFbugFMYhTw6PBVJ8?usp=drive_link), and [mead.pth](https://drive.google.com/drive/folders/1IYVyYp35ueNbXZ3XFbugFMYhTw6PBVJ8?usp=drive_link). Put the pretrained models under `BIWI`, `vocaset`, and `mead` folders, respectively. Given the audio signal,

- to animate a mesh in VOCASET topology, run: 
	```
	python demo.py vocaset --audio_file <audio_path>
	```
- to animate a mesh in BIWI topology, run: 
	```
	python demo.py BIWI --audio_file <audio_path>
	```
- to animate a mesh in 3D-MEAD topology, run: 
	```
	python demo.py MEAD --audio_file <audio_path>
	```
	This script will automatically generate the rendered videos in the `demo/output` folder. 

## **Training / Testing**

The training/testing operation shares a similar command:
```
python main.py <vocaset|BIWI|MEAD>
```
After the above statement is executed, the test is automatically performed after the training is completed, if you want to test separately, you can use the following command:
```
python test.py <vocaset|BIWI|MEAD>
```

## **Citation**
If you find our code or paper useful, please consider citing
```bibtex
@ARTICLE{talkingstyle,
  author={Song, Wenfeng and Wang, Xuan and Zheng, Shi and Li, Shuai and Hao, Aimin and Hou, Xia},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={TalkingStyle: Personalized Speech-Driven 3D Facial Animation with Style Preservation}, 
  year={2024},
  pages={1-12},}
```


## **Acknowledgement**
We heavily borrow the code from
[FaceFormer](https://github.com/EvelynFan/FaceFormer),
[CodeTalker](https://github.com/RenYurui/PIRender), and
[VOCA](https://github.com/TimoBolkart/voca). Thanks
for sharing their code and [huggingface-transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py) for their HuBERT implementation. We also gratefully acknowledge the ETHZ-CVL for providing the [B3D(AC)2](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset and MPI-IS for releasing the [VOCASET](https://voca.is.tue.mpg.de/) dataset. Any third-party packages are owned by their respective authors and must be used under their respective licenses.


