### triton vs cuda

#### 特性

+ triton不需要考虑shared memory，而cuda需要手动实现

#### program == block

> Note on jargon: In triton lingo, each kernel (which processes a block) is called a "program". Therefore, "block_id" is often called "pid" (short for "program id"), but it's the same.

+ triton中的program就是cuda中的block
+ cuda把计算分成block和thread，而triton只分为block。
  triton在block中可以直接进行vector运算，因此比cuda好写得多

#### triton kernel中的操作

> **All** operations in triton kernels are vectorized: Loading data, operating on data, storing data, and creating masks.

+ triton kernel中一共只有四种操作：存/取数据、计算、越界检测

#### 什么是grid

+ 一个tensor
+ 用来指定block(program)的分块维度，每个维度分成多少块



------------------------

### Matrix Add

#### 图解

![image-20240909205314939](C:\Users\Ganzeus\AppData\Roaming\Typora\typora-user-images\image-20240909205314939.png)

#### 代码

![image-20240909210039363](C:\Users\Ganzeus\AppData\Roaming\Typora\typora-user-images\image-20240909210039363.png)

### Matmul

#### 图解

![image-20240923234331822](C:\Users\Ganzeus\AppData\Roaming\Typora\typora-user-images\image-20240923234331822.png)


#### 代码

![image-20240923234608376](C:\Users\Ganzeus\AppData\Roaming\Typora\typora-user-images\image-20240923234608376.png)

![image-20240923234640128](C:\Users\Ganzeus\AppData\Roaming\Typora\typora-user-images\image-20240923234640128.png)
