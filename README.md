## Preprocessing
```
pip install -r requirements.txt
```
#### Dataset
```
BDD100K: (https://bdd-data.berkeley.edu)
Tusimple: (https://github.com/TuSimple/tusimple-benchmark/issues/3)
Culane: (https://xingangpan.github.io/projects/CULane.html)
```
#### weights
weight download link：https://drive.google.com/file/d/1s47pGa5pTC3W_10-TwSxGXKdBmXxar4k/view?usp=drive_link

## Detect
```
python tools/test.py --weights weights/HFFMTransformer.pth
```
## Acknowledgments
我们主要参考如下工作，感谢他们精彩的工作。YOLOP(https://github.com/hustvl/YOLOP)
Multinet(https://github.com/MarvinTeichmann/MultiNet)
