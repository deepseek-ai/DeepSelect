# KernelKit

## Overview

KernelKit is a collection of tools and utilities for testing & benchmarking kernels. Currently it includes the following submodules:

- `kernelkit.compare`: Various functions for comparing two tensors.
- `kernelkit.benchmark`: Functions for benchmarking kernel performance.
- `kernelkit.utils`: Utility functions for kernel testing and benchmarking.

## Usage

首先先明确两个基本概念：

- 下文中的“测试用例配置”一词，指的是一组参数，用来描述一个测试用例的属性，比如 batch size, sequence length 之类的，其大小一般很小。你也可以叫它“训练超参数”。
- 下文中的“测试用例”一词，指的是一个具体的测试用例实例，包含若干个 `torch.Tensor`，其大小可能较大。

本项目**不是**一个 Python distribution package，仅仅是一个 source module。通俗地说，意思就是，这个项目没法通过 `pip install` 或者 `setup.py` 之类的方式安装，而是需要 git clone 成 submodule 之后，直接 import 使用。

对于正确性测试以及性能评测，直接使用 `kernelkit.compare` 和 `kernelkit.benchmark` 中提供的函数即可，可以参考 [FlashMLA 中的脚本](https://gitlab.deepseek.com/deepseek/flash-mla/-/blob/main/tests/test_flash_mla.py?ref_type=heads)（关注那些 `kk.` 开头的函数）。

