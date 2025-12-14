#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 简化测试脚本，直接调用修复后的代码

import sys
import os
import re
import numpy as np

# 导入修复后的类定义
from analyze_abinit_bands-3 import AbinitBandAnalyzer

def test_parser():
    """测试AGR文件解析"""
    print("开始测试AGR文件解析...")
    
    # 创建分析器实例
    analyzer = AbinitBandAnalyzer('nacl-scf-bando_DS2_EBANDS.agr', 'nacl-scf-band.abi')
    
    # 解析AGR文件
    result = analyzer.parse_agr_file()
    print(f"AGR文件解析结果: {result}")
    
    # 打印解析结果
    print(f"找到 {analyzer.num_kpoints} 个k点")
    print(f"找到 {analyzer.num_bands} 条能带")
    
    if analyzer.high_sym_points:
        print(f"找到高对称点: {analyzer.high_sym_points}")
    
    # 测试k点路径计算
    k_path = analyzer.calculate_k_path_coordinates()
    print(f"k路径计算结果: {k_path}")
    
    return result

if __name__ == "__main__":
    print("测试修复后的ABINIT能带分析脚本...")
    
    try:
        result = test_parser()
        if result:
            print("\n✓ 测试通过！AGR文件解析成功")
        else:
            print("\n✗ 测试失败！AGR文件解析失败")
    except Exception as e:
        print(f"\n✗ 测试失败！发生异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成！")