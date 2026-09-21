# 一、imshow灰度图
1. `plt.imshow(img, cmap="gray", vmin=0, vmax=30000)` 图像 灰色 显示范围 数值根据像素值来取 ，如果（... vmin=50, vmax=200）那么0，50，50 都显示以一个颜色 200，200，255都显示一个颜色
```
img = np.array([

    [0, 50, 100],

    [50, 150, 200],

    [100, 200, 255]

])
```
# 二、直方图
1. **作用**：直观的显示这张图的**像素值怎么分布**
2. `plt.hist(img.ravel(), bins=256)` 平铺img[[Numpy 基础#三、形状操作]]，分成256个区间
3. 可以观察：图像**整体明、暗**；有没有大量像素**接近 0**；**噪声**大致分布；**归一化**以后的范围
# 三、子图
1. 作用：同时显示多个子图，方便对比
2. `fig, (ax1, ax2) = plt.subplots(1, 2)` 或者使用`fig, axes = plt.subplots(1, 2)`，结果一样但是后面的用法不同
3. 第一种方式直接使用ax1 ax2做操作，比如`ax1.imshow(img,cmap="gray")`画灰度图
4. 第二种方式是使用数组索引操作，axes实际上是一个数组 元素个数取决与画布创建的形状 这里就是2个 使用`axes[0].hist(img.ravel(),bins=128)`画直方图
# 四、colorbar
1. **作用**：现在这个亮度到底代表多少像素值
2. 
```
im = plt.imshow(img, cmap="gray")
plt.colorbar(im)
plt.show()
```





