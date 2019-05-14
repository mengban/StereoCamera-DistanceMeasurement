### 需求:使用双目摄像头得出物体3D坐标，本质就是利用双目来得到深度信息。

#### 0 知识扫盲
- [相机模型](https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html)
- [四大坐标关系及其关系](https://blog.csdn.net/waeceo/article/details/50580607)
- []()

#### 1 相机标定
- Q1:用MATLAB标定还是opencv标定？
- A1:两种我都试了。总结来说，直接影响标定结果的好坏的因素是图片质量，值图片质量较好的情况下，两者结果基本一样。

- Q2:是两个相机一起标定还是单独标定?
- A2:MATLAB 和 OPENCV中都有单目标定和双目标定(MATLAB版本>2014)的方式。题主采用的方案是opencv分开标， MATLAB一起标。opencv分开标的主要原因是利用opencv **cv2.stereoCalibrate()**标出的两相机间的RT矩阵实在偏差太大，所以采用了分开标定相机。而MATLAB计算的结果就相当好，示意图和我实际摆放的相机位置基本一样。

> 使用MATLAB标定。左右照片各15张(共采集19张，MATLAB识别出有效16张，手动删除一张Mean Erro较大的图)。记下 内参参数及两相机间的RT矩阵。MATLAB 标定结果如下。设置棋盘格单位长度25mm。
![](https://raw.githubusercontent.com/mengban/ImageHosting/master/cnblog/matlab1.png)
可以看出标定出的的结果相机及棋盘的摆放位置与实际摆放接近。
![](https://raw.githubusercontent.com/mengban/ImageHosting/master/cnblog/matlab2.png)
图中标定显示棋盘与相机大概700~800mm的距离(棋盘格单位为25mm的前提下)。
#### 2 计算3d坐标。
题主主要利用了opencv提供的
[cv2.triangulatePoints()函数](https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#gad3fc9a0c82b08df034234979960b778c)

``` python
'''
参数含义:
projMatr1	3x4 projection matrix of the first camera.
projMatr2	3x4 projection matrix of the second camera.
projPoints1	2xN array of feature points in the first image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
projPoints2	2xN array of corresponding points in the second image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
points4D	4xN array of reconstructed points in homogeneous coordinates.
'''
points4D = cv.triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D])


```

可见关键步是计算出两个投影矩阵， 然后将待测物体在左右相机成像的像素坐标代入即可得到3d坐标。原理、方法都很简单。
投影矩阵的计算有两种方式:
- 1.采用立体标定的方案(opencv)。
主要思路是cv2.stereoCalibrate()计算出R|T，将其代入cv2.stereoRectify()得到投影矩阵P1，P2。
此时得到的P1 P2为:
![](https://raw.githubusercontent.com/mengban/ImageHosting/master/cnblog/P1P2.png)
P1 和P2 之间只差一个平移矩阵。与我们实际摆放的位置不符。实际摆放位置和MATLAB标定的结果类似。原因不清楚，然后采用了方案2计算投影矩阵。

- 2 自己计算投影矩阵。即内参与外参的乘积。
用图片/stereo512/left.bmp /stereo512/right.bmp计算相机外参。使用了[cv.solvePnP()函数](https://docs.opencv.org/4.1.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)

#### 3.验证.
>将棋盘格左右像素坐标代入函数cv.triangulatePoints()得到棋盘格格点3d坐标.
**MATLAB标定结果**:
![](https://raw.githubusercontent.com/mengban/ImageHosting/master/cnblog/p3d.png)
**opencv标定结果**:
![](https://raw.githubusercontent.com/mengban/ImageHosting/master/cnblog/p3d-opencv.png)
可以看出在3维坐标的计算上，三个轴3d坐标与实际值相差都很小，并且opencv标注产生的均方误差在三个轴均略优于MATLAB。

>继续验证:
 