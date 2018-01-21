在运行卷积操作和池化操作的快速版本时需要在当前目录运行如下命令:
python setup.py build_ext --inplace

该命令需要使用python3.5，而本机上python默认版本是2.7，所以需要切换到conda中安装的python3.5

source activate py3

然后再运行上面的命令。
如果运行时出现:global name ‘col2im_6d_cython’ is not defined 错误，则重新启动jupyter即可。
