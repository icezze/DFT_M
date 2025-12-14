#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"命令行参数: {sys.argv}")

# 测试文件读取
try:
    with open('nacl-scf-bando_DS2_EBANDS.agr', 'r') as f:
        lines = f.readlines()
    print(f"成功读取文件，共 {len(lines)} 行")
except Exception as e:
    print(f"读取文件失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成！")