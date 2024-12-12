# trajdl

这是一个正在开发中的，用于轨迹深度学习的算法包。

## 开发环境安装

python>=3.8，先按照pytorch的文档安装好pytorch，最低版本2.1.0

```bash
# 设置环境变量
source scripts/build/pre.sh

# 开发环境构建，测试用例测试，wheel包打包
bash scripts/build/dev.sh
```

docker镜像：
- CPU：docker/cpu.dockerfile
- GPU：docker/gpu.dockerfile

## Benchmark

`scripts/benchmark`目录下存储了`TrajDL`提供的benchmark，针对各个论文使用`TrajDL`进行了复现。

## License
本项目使用**Apache License 2.0**，详见[LICENSE](LICENSE)。

## 如何编写文档？

安装jupyter相关的工具
`pip install jupyter jupytext jupyterlab_myst`

jupyter是用来启动jupyter lab的
jupytext是用来做`notebook <--> md`转换用的
jupyterlab_myst是用来在jupyter里面渲染`myst markdown`的

启动`jupyter lab`，然后进入jupyter的网页，进入docs目录，找到要编辑的markdown文件，右键点击，选择`Open With` -> `Notebook`

编辑完成后保存即可，然后正常用命令行编译文档即可。