## MMDetection
Fork from https://github.com/open-mmlab/mmdetection

## 安装虚拟环境
参考 https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md
```bash
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
```

## 数据准备
目的是转换成COCO格式的数据，方式有两：
1. 下载: [百度云链接](https://pan.baidu.com/s/1dUNt_Wl3WVbqip6oVQ_tuA)  密码: oqqf
2. 自己生成(请合理配置文件中的路径):
```bash
# 文件中的路径请根据实际情况配置
# 训练集与测试集的划分
python tools/pascal2coco.py

# a-test/b-test转换成COCO json
python tools/generate_test_annotation.py
```
3. 最后的目录树
```bash
data
| - image # 前视, 侧扫, 负样本, 所有的训练集图像均放于此
| - annotation # 上一步下载或者生成的json
| - box # 前视, 侧扫的xml标注放于此
| - a-test-img # 所有的a-test图像放于此
| - b-test-img # 所有的b-test图像放于此
```

## 模型准备
放于`pretrain_model/`下
1. 预训练模型：下载[百度云链接](https://pan.baidu.com/s/1Kadwx588xZmy6qM_1cgdBw) 密码: o22i (已在MSCOCO2017上预训练)，
2. 最终已训练模型： 下载[百度云链接](https://pan.baidu.com/s/1LoXNvHmk4O3yVUrP3ix6HQ) 密码: v15d (final model)

## 训练
```bash
./train_sonar.sh # 目录位置及配置可在里面更改
```
## 测试
```bash
./test_sonar.sh # 目录位置及配置可在里面更改

# 转换成csv
python ./tools/turn_pkl_to_csv.py work_dirs/cascade_rcnn_dconv_c3-c5_r101_uncertain_MS_dufpn_1x/bbox_predict.pkl --json_path data/annotation/a-test.json

# 可视化, 目录位置可在里面更改
python tools/vis_det.py 
```

## 结果

| -   | Leaderboard A | Leaderboard B |  
| :---: | :-------------: | :-------------: |  
| mAP | 0.298         | 0.207         |  


