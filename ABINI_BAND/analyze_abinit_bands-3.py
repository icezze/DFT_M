#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABINIT能带数据分析工具 - 增强版
支持多种ABINIT输出格式，自动处理高对称点路径
用法: python abinit_band_analyzer.py <agr文件> [--input-file <abi文件>]
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import argparse
from pathlib import Path

class AbinitBandAnalyzer:
    def __init__(self, agr_file, abi_file=None):
        """
        初始化分析器
        
        Parameters:
        -----------
        agr_file : str
            AGR格式的能带文件路径
        abi_file : str, optional
            ABINIT输入文件路径（用于获取晶格和高对称点信息）
        """
        self.agr_file = agr_file
        self.abi_file = abi_file
        self.lattice_constant = None  # 晶格常数 (Å)
        self.rprim = None  # 原胞基矢
        self.k_points = []  # k点分数坐标
        self.band_data = []  # 能带数据
        self.high_sym_points = {}  # 从AGR文件中提取的高对称点信息
        self.high_sym_path = []  # 从输入文件中提取的高对称点路径
        self.fermi_energy = 0.0  # 费米能级 (eV)
        self.num_bands = 0
        self.num_kpoints = 0
        self.efermi_original = None  # 原始费米能级
        self.band_gap_info = None  # 带隙分析结果
        self.k_path = None  # k路径坐标
        self.k_path_length = None  # 归一化的k路径长度
        self.enunit2 = None  # 能量单位设置 (1 = eV, 0 = Hartree)
        self.agr_energy_zero_is_efermi = False  # 标记AGR文件中的能量零点是否已经是费米能级
        
    def parse_agr_file(self):
        """解析AGR格式的文件（支持多种格式）"""
        print(f"正在解析AGR文件: {self.agr_file}...")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"文件是否存在: {os.path.exists(self.agr_file)}")
        if os.path.exists(self.agr_file):
            print(f"文件大小: {os.path.getsize(self.agr_file)} bytes")
        
        try:
            with open(self.agr_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"成功读取文件，共 {len(lines)} 行")
        except FileNotFoundError:
            print(f"错误: 文件不存在 - {self.agr_file}")
            sys.exit(1)
        except Exception as e:
            print(f"错误: 无法读取文件 - {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # 尝试提取晶格常数（不同格式）
        for line in lines:
            if 'abc' in line and ':' in line:  # 格式1: abc   :   3.814814   3.814814   3.814814
                parts = line.split(':')
                if len(parts) > 1:
                    lattice_str = parts[1].strip()
                    # 提取第一个晶格常数（对于立方晶系，a=b=c）
                    try:
                        self.lattice_constant = float(lattice_str.split()[0])
                        print(f"晶格常数: {self.lattice_constant} Å")
                    except:
                        pass
                break
            elif 'acell' in line:  # 可能在注释中有acell信息
                match = re.search(r'acell\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', line)
                if match:
                    try:
                        self.lattice_constant = float(match.group(1))
                        print(f"从注释中找到晶格常数: {self.lattice_constant} Å")
                    except:
                        pass
        
        # 提取原始费米能级
        efermi_found = False
        self.agr_energy_zero_is_efermi = False  # 标记AGR文件中的能量零点是否已经是费米能级
        
        for line in lines:
            if 'Zero set to efermi' in line or 'previously it was at:' in line:
                matches = re.findall(r'[-+]?\d*\.\d+[Ee]?[-+]?\d*', line)
                if matches:
                    try:
                        self.efermi_original = float(matches[0])
                        print(f"原始费米能级: {self.efermi_original} eV")
                        efermi_found = True
                        self.agr_energy_zero_is_efermi = True
                    except:
                        pass
                break
        
        # 如果没有找到费米能级，使用默认值
        if not efermi_found:
            self.efermi_original = 0.0
            print(f"未找到原始费米能级，使用默认值: {self.efermi_original} eV")
        
        # 提取k点坐标 - 支持多种格式
        in_kpoints_section = False
        kpoints_started = False
        
        # 查找k点列表的行范围
        kpoints_start_idx = -1
        kpoints_end_idx = -1
        
        for i, line in enumerate(lines):
            if 'List of k-points' in line or 'k-points and their index' in line:
                kpoints_start_idx = i + 1
                break
        
        if kpoints_start_idx != -1:
            for i in range(kpoints_start_idx, len(lines)):
                line = lines[i]
                if line.strip() and not line.strip().startswith('#'):
                    kpoints_end_idx = i
                    break
        
        # 解析k点列表
        if kpoints_start_idx != -1 and kpoints_end_idx != -1:
            for line in lines[kpoints_start_idx:kpoints_end_idx]:
                if line.strip().startswith('#'):
                    # 解析k点行，支持多种格式:
                    # 格式1: "# 0 [0.5 0.  0. ]"
                    # 格式2: "# 0 [ 0.0000E+00,  0.0000E+00,  0.0000E+00]"
                    # 格式3: "# 0 0.5 0.0 0.0" (不常见)
                    
                    # 尝试格式1和格式2
                    match1 = re.search(r'#\s*(\d+)\s*\[([\d\.\s\-,E+]+)\]', line)
                    if match1:
                        k_index = int(match1.group(1))
                        coords_str = match1.group(2)
                        # 清理坐标字符串，移除逗号，处理科学计数法
                        coords_str = coords_str.replace(',', ' ')
                        # 将科学计数法的E替换为e以便Python解析
                        coords_str = coords_str.replace('E', 'e')
                        # 提取三个坐标值
                        coords = []
                        for x in coords_str.split():
                            try:
                                coords.append(float(x))
                            except:
                                pass
                        if len(coords) >= 3:
                            self.k_points.append(coords[:3])
                    else:
                        # 尝试其他格式
                        match2 = re.search(r'#\s*(\d+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)', line)
                        if match2:
                            k_index = int(match2.group(1))
                            coords = [float(match2.group(2)), float(match2.group(3)), float(match2.group(4))]
                            self.k_points.append(coords)
        
        # 如果没有从AGR文件中找到k点列表，尝试从能带数据中推断k点数量
        if not self.k_points:
            print("未找到k点列表，将从能带数据中推断k点数量")
            # 先解析能带数据，获取k点数量
            temp_bands = []
            temp_band = []
            current_band = -1
            
            for line in lines:
                if '@target G0.S' in line:
                    if temp_band and current_band >= 0:
                        temp_bands.append(temp_band)
                        temp_band = []
                    match = re.search(r'S(\d+)', line)
                    if match:
                        current_band = int(match.group(1))
                elif line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-' or line.strip()[0] == '.'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            energy = float(parts[1])
                            temp_band.append(energy)
                        except:
                            continue
            
            if temp_band and current_band >= 0:
                temp_bands.append(temp_band)
            
            if temp_bands:
                self.num_kpoints = len(temp_bands[0])
                print(f"从能带数据推断k点数量: {self.num_kpoints}")
                # 生成默认的k点坐标（0到1之间的均匀分布）
                for i in range(self.num_kpoints):
                    # 生成简单的分数坐标，假设是线性路径
                    self.k_points.append([i/(self.num_kpoints-1) if self.num_kpoints > 1 else 0.0, 0.0, 0.0])
        else:
            self.num_kpoints = len(self.k_points)
        
        print(f"找到/推断 {self.num_kpoints} 个k点")
        
        # 提取高对称点信息（从AGR文件的xaxis标记中）
        for line in lines:
            if '@xaxis  tick major' in line and 'ticklabel' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    # 格式: @xaxis  tick major 0, "L"
                    try:
                        # 提取坐标和标签
                        coord_part = parts[0].split()
                        for item in coord_part:
                            if item.replace('.', '').isdigit():
                                coord = float(item)
                                break
                        label = parts[1]
                        self.high_sym_points[label] = coord
                    except:
                        continue
            elif '@xaxis  tick major' in line and not 'ticklabel' in line:
                # 有些文件只有tick major位置，没有标签
                parts = line.split()
                try:
                    for i, part in enumerate(parts):
                        if part == 'major' and i+2 < len(parts):
                            coord = float(parts[i+2].replace(',', ''))
                            # 如果没有标签，使用默认标签
                            if coord not in self.high_sym_points.values():
                                self.high_sym_points[f"Point_{int(coord)}"] = coord
                except:
                    pass
        
        # 提取能带数据
        current_band = -1
        band_values = []
        
        for line in lines:
            # 检测新的能带数据开始
            if '@target G0.S' in line:
                if band_values and current_band >= 0:
                    self.band_data.append(band_values)
                    band_values = []
                # 提取能带编号
                match = re.search(r'S(\d+)', line)
                if match:
                    current_band = int(match.group(1))
            
            # 解析数据行（数字开头）
            elif line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-' or line.strip()[0] == '.'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # 第一个值是k点索引或x坐标，第二个是能量值
                        energy = float(parts[1])
                        band_values.append(energy)
                    except:
                        continue
        
        # 添加最后一个能带的数据
        if band_values and current_band >= 0:
            self.band_data.append(band_values)
        
        self.num_bands = len(self.band_data)
        print(f"找到 {self.num_bands} 条能带，每条 {len(self.band_data[0]) if self.band_data else 0} 个点")
        
        # 验证数据一致性
        if self.band_data:
            for i, band in enumerate(self.band_data):
                if len(band) != self.num_kpoints:
                    print(f"警告: 能带 {i} 有 {len(band)} 个点，但期望 {self.num_kpoints} 个点")
        
        # 计算k路径坐标
        self.k_path = self.calculate_k_path_coordinates()
        print(f"k_path类型: {type(self.k_path)}")
        if self.k_path and len(self.k_path) == 2:
            print(f"k_path[0]长度: {len(self.k_path[0])}, k_path[1]长度: {len(self.k_path[1])}")
        
        return True
    
    def parse_abo_file(self, abo_file):
        """解析ABINIT输出文件，获取额外信息（不再提取费米能级）"""
        print(f"正在解析ABINIT输出文件: {abo_file}")
        
        try:
            with open(abo_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误: 无法读取输出文件 - {e}")
            return False
        
        # 从ABO文件中提取Γ点的价带顶和导带底能量（仅用于信息展示）
        # 匹配格式: kpt#  11, nband=  8, wtk=  1.00000, kpt=  0.0000  0.0000  0.0000 (reduced coord)
        # -2.6548E-01  1.7916E-01  1.7916E-01  1.7916E-01  2.7274E-01  2.7274E-01  2.7274E-01  3.0388E-01
        gamma_match = re.search(r'kpt#\s+\d+,\s+nband=\s+\d+,\s+wtk=\s+\d+\.\d+,\s+kpt=\s+0\.0000\s+0\.0000\s+0\.0000\s+\(reduced coord\)\s*\n\s*([-+\d\.\sEe]+)', content)
        if gamma_match:
            gamma_energies_str = gamma_match.group(1)
            gamma_energies = list(map(float, gamma_energies_str.split()))
            print(f"Γ点能量值: {gamma_energies}")
            
            # 检查这些能量值是否与AGR文件中的值一致
            if self.band_data and len(self.band_data) >= 5:
                agr_gamma_energies = [band[0] for band in self.band_data]
                print(f"AGR文件中Γ点能量值: {agr_gamma_energies[:8]}")
        
        return True
    
    def parse_abi_file(self):
        """解析ABINIT输入文件，获取晶体结构和高对称点路径（不获取费米能级）"""
        if not self.abi_file or not os.path.exists(self.abi_file):
            print("警告: 未提供ABINIT输入文件或文件不存在")
            return False
        
        print(f"正在解析ABINIT输入文件: {self.abi_file}")
        
        try:
            with open(self.abi_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"错误: 无法读取输入文件 - {e}")
            return False
        
        lines = content.split('\n')
        
        # 提取晶格常数
        # 支持多种格式: acell 3*10.195 或 acell 10.195 10.195 10.195
        acell_value = None
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            # 匹配 acell 3*10.195 格式
            acell_match = re.search(r'acell\s+3\*([\d\.]+)', line_stripped)
            if acell_match:
                try:
                    acell_value = float(acell_match.group(1))
                    break
                except:
                    pass
            
            # 匹配 acell 10.195 10.195 10.195 格式
            acell_match = re.search(r'acell\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', line_stripped)
            if acell_match:
                try:
                    acell_value = float(acell_match.group(1))
                    break
                except:
                    pass
        
        if acell_value is not None:
            # 通常acell以Bohr为单位，需要转换为Å（1 Bohr = 0.529177210903 Å）
            bohr_to_ang = 0.529177210903
            self.lattice_constant = acell_value * bohr_to_ang
            print(f"晶格常数: {self.lattice_constant} Å (转换自 {acell_value} Bohr)")
        
        # 提取原胞基矢
        rprim_values = []
        rprim_found = False
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            if 'rprim' in line_stripped and not rprim_found:
                rprim_found = True
                # 提取当前行的rprim值
                parts = line_stripped.split()[1:]  # 跳过 'rprim' 关键字
                rprim_values.extend([float(x) for x in parts if x.replace('.', '').replace('-', '').isdigit()])
                continue
            
            if rprim_found:
                # 继续提取后续行的rprim值
                if len(rprim_values) >= 9:
                    break
                
                # 检查是否是新的变量行
                if ':' in line_stripped or any(keyword in line_stripped for keyword in ['ntypat', 'znucl', 'natom', 'typat', 'xred', 'xcart', 'ecut', 'kptopt']):
                    break
                
                # 提取当前行的rprim值
                parts = line_stripped.split()
                rprim_values.extend([float(x) for x in parts if x.replace('.', '').replace('-', '').isdigit()])
        
        if len(rprim_values) >= 9:
            self.rprim = np.array(rprim_values[:9]).reshape(3, 3)
            print(f"找到原胞基矢")
        
        # 提取高对称点路径 - 支持多种数据集和kptbounds格式
        kpt_points = []
        
        # 支持多个数据集的kptbounds（kptbounds1, kptbounds2, ...）
        for dataset_num in range(1, 11):  # 最多检查10个数据集
            kptbounds_keyword = f'kptbounds{dataset_num}'
            in_kptbounds = False
            kptbounds_start = False
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if kptbounds_keyword in line_stripped and not line_stripped.startswith('#'):
                    in_kptbounds = True
                    continue
                
                if in_kptbounds:
                    # 跳过注释行
                    if line_stripped.startswith('#'):
                        continue
                    
                    # 检查是否为空行或新变量开始
                    if not line_stripped:
                        break
                    
                    # 检查是否是数字行（高对称点坐标）
                    parts = line_stripped.split()
                    if len(parts) >= 3:
                        try:
                            # 尝试解析三个坐标
                            coords = [float(parts[0]), float(parts[1]), float(parts[2])]
                            
                            # 提取标签（如果有）
                            label = None
                            if '#' in line:
                                label_part = line.split('#')[1].strip()
                                # 移除可能的$符号和转义字符
                                label_part = label_part.replace('$', '').replace('\\', '').strip()
                                if label_part:
                                    label = label_part
                            
                            # 标准化标签
                            if label:
                                # 替换常见的标签名称
                                label = label.replace('Gamma', 'Γ')
                                label = label.replace('gamma', 'Γ')
                            
                            kpt_points.append({
                                'coords': coords,
                                'label': label or f"Point_{len(kpt_points)}"
                            })
                            kptbounds_start = True
                        except:
                            # 如果不是数字，可能是下一个变量
                            if kptbounds_start:
                                break
                    elif kptbounds_start:
                        # 已经开始了但遇到非数字行，结束
                        break
            
            if kpt_points:
                break  # 找到第一个数据集的kptbounds就退出
        
        if kpt_points:
            self.high_sym_path = kpt_points
            print(f"从输入文件中找到 {len(kpt_points)} 个高对称点")
            for i, point in enumerate(kpt_points):
                print(f"  点 {i}: {point['coords']} - {point['label']}")
        
        # 提取能带数量
        nband = None
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            # 支持多个数据集的nband（nband1, nband2, ...）
            nband_match = re.search(r'nband(\d*)\s+(\d+)', line_stripped)
            if nband_match:
                try:
                    nband = int(nband_match.group(2))
                    print(f"从输入文件中找到能带数量: {nband}")
                    break
                except:
                    pass
        
        # 提取能量单位设置（enunit2）
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            enunit2_match = re.search(r'enunit2\s+(\d+)', line_stripped)
            if enunit2_match:
                try:
                    self.enunit2 = int(enunit2_match.group(1))
                    print(f"从输入文件中找到能量单位设置enunit2: {self.enunit2}")
                    break
                except:
                    pass
        
        return True
    
    def calculate_reciprocal_lattice(self):
        """计算倒易空间晶格"""
        if not self.lattice_constant and self.rprim is None:
            print("警告: 未找到晶格信息，使用默认值估算")
            # 尝试从k点坐标估算
            if self.k_points:
                # 假设是立方晶系，估算晶格常数
                # 寻找最大的k点坐标差异
                max_coord = max([max(abs(c) for c in k) for k in self.k_points])
                if max_coord > 0:
                    # 假设最大k点坐标对应布里渊区边界
                    self.lattice_constant = 2 * np.pi / max_coord
                    print(f"估算晶格常数: {self.lattice_constant:.4f} Å")
        
        if self.lattice_constant:
            a = self.lattice_constant
            print(f"\n晶格常数: {a} Å")
            
            # 对于立方晶系，简化计算
            # 倒格矢: b_i = 2π/a
            b_mag = 2 * np.pi / a
            
            print(f"\n倒格子基矢大小 (单位: 2π/Å):")
            print(f"|b| = {b_mag:.6f}")
            
            return b_mag
        
        elif self.rprim is not None:
            print(f"\n原胞基矢:")
            print(f"a₁ = [{self.rprim[0,0]:.6f}, {self.rprim[0,1]:.6f}, {self.rprim[0,2]:.6f}]")
            print(f"a₂ = [{self.rprim[1,0]:.6f}, {self.rprim[1,1]:.6f}, {self.rprim[1,2]:.6f}]")
            print(f"a₃ = [{self.rprim[2,0]:.6f}, {self.rprim[2,1]:.6f}, {self.rprim[2,2]:.6f}]")
            
            # 计算原胞体积
            volume = np.abs(np.linalg.det(self.rprim))
            print(f"原胞体积: {volume:.6f} Å³")
            
            # 计算倒格子基矢
            b1 = 2 * np.pi * np.cross(self.rprim[1], self.rprim[2]) / volume
            b2 = 2 * np.pi * np.cross(self.rprim[2], self.rprim[0]) / volume
            b3 = 2 * np.pi * np.cross(self.rprim[0], self.rprim[1]) / volume
            
            print(f"\n倒格子基矢 (单位: 2π/Å):")
            print(f"b₁ = [{b1[0]:.6f}, {b1[1]:.6f}, {b1[2]:.6f}]")
            print(f"b₂ = [{b2[0]:.6f}, {b2[1]:.6f}, {b2[2]:.6f}]")
            print(f"b₃ = [{b3[0]:.6f}, {b3[1]:.6f}, {b3[2]:.6f}]")
            
            return b1, b2, b3
        
        return None
    
    def get_k_point_symbol(self, k_coords, tolerance=1e-4):
        """判断k点是否在高对称点上，返回高对称点符号"""
        # 首先检查从输入文件解析的高对称点路径
        if self.high_sym_path:
            for point in self.high_sym_path:
                ref_coords = point['coords']
                if (abs(k_coords[0] - ref_coords[0]) < tolerance and
                    abs(k_coords[1] - ref_coords[1]) < tolerance and
                    abs(k_coords[2] - ref_coords[2]) < tolerance):
                    return point['label']
        
        # 检查常见的高对称点
        common_sym_points = {
            (0.0, 0.0, 0.0): "Γ",
            (0.5, 0.0, 0.5): "X",
            (0.5, 0.5, 0.5): "L",
            (0.5, 0.25, 0.75): "W",
            (0.375, 0.375, 0.75): "K",
            (0.625, 0.25, 0.625): "U",
        }
        
        for sym_coords, symbol in common_sym_points.items():
            if (abs(k_coords[0] - sym_coords[0]) < tolerance and
                abs(k_coords[1] - sym_coords[1]) < tolerance and
                abs(k_coords[2] - sym_coords[2]) < tolerance):
                return symbol
        
        return None
    
    def calculate_k_path_coordinates(self):
        """计算k路径的实际坐标（考虑倒易空间距离）"""
        if not self.k_points:
            print("错误: 未找到k点数据")
            return np.zeros(1), np.zeros(1)
        
        # 尝试使用晶格信息计算更精确的路径长度
        if self.rprim is not None:
            # 有原胞基矢，计算倒格矢
            volume = np.abs(np.linalg.det(self.rprim))
            b1 = 2 * np.pi * np.cross(self.rprim[1], self.rprim[2]) / volume
            b2 = 2 * np.pi * np.cross(self.rprim[2], self.rprim[0]) / volume
            b3 = 2 * np.pi * np.cross(self.rprim[0], self.rprim[1]) / volume
            
            # 计算每个k点在倒易空间中的直角坐标
            k_cartesian = []
            for k_frac in self.k_points:
                k_cart = k_frac[0] * b1 + k_frac[1] * b2 + k_frac[2] * b3
                k_cartesian.append(k_cart)
        elif self.lattice_constant:
            # 立方晶系简化计算
            a = self.lattice_constant
            b_mag = 2 * np.pi / a
            
            # 对于立方晶系，简化处理
            k_cartesian = []
            for k_frac in self.k_points:
                # 近似处理
                k_cart = np.array(k_frac) * b_mag
                k_cartesian.append(k_cart)
        else:
            # 没有晶格信息，使用分数坐标的欧几里得距离
            k_cartesian = [np.array(k) for k in self.k_points]
        
        # 计算路径长度（累加直线距离）
        k_path_coords = np.zeros(self.num_kpoints)
        
        for i in range(1, self.num_kpoints):
            # 计算相邻k点间的距离
            distance = np.linalg.norm(k_cartesian[i] - k_cartesian[i-1])
            k_path_coords[i] = k_path_coords[i-1] + distance
        
        # 归一化到0-1范围，便于绘图
        if k_path_coords[-1] > 0:
            k_path_normalized = k_path_coords / k_path_coords[-1]
        else:
            k_path_normalized = k_path_coords
        
        return k_path_coords, k_path_normalized
    
    def find_high_symmetry_indices(self):
        """在k点列表中查找高对称点的索引"""
        if not self.high_sym_path or not self.k_points:
            return {}
        
        high_sym_indices = {}
        tolerance = 1e-4
        
        for point in self.high_sym_path:
            point_coords = point['coords']
            label = point['label']
            
            # 在k点列表中查找匹配的坐标
            for i, k_coords in enumerate(self.k_points):
                if (abs(k_coords[0] - point_coords[0]) < tolerance and
                    abs(k_coords[1] - point_coords[1]) < tolerance and
                    abs(k_coords[2] - point_coords[2]) < tolerance):
                    
                    if label not in high_sym_indices:
                        high_sym_indices[label] = i
                        print(f"高对称点 {label} 对应k点索引: {i}")
                    break
        
        # 如果没有从输入文件找到高对称点，尝试从AGR文件中提取
        if not high_sym_indices and self.high_sym_points:
            for label, coord in self.high_sym_points.items():
                idx = int(coord)
                if 0 <= idx < len(self.k_path):
                    high_sym_indices[label] = idx
        
        return high_sym_indices
    
    def analyze_band_gap(self, interactive_mode=False):
        """
        分析带隙信息
        """
        print("\n=== 开始带隙分析 ===")
        print(f"当前费米能级: {self.fermi_energy} eV")
        print(f"原始费米能级: {self.efermi_original} eV")
        
        if not self.band_data:
            print("错误: 没有能带数据")
            return None
        
        if self.k_path is None:
            print("错误: 未计算k路径坐标")
            return None
        
        # 1. 分析每条能带的行为
        print("\n1. 分析每条能带的行为:")
        band_behavior = []
        for band_idx in range(self.num_bands):
            behavior = self._analyze_single_band(band_idx)
            band_behavior.append(behavior)
            print(f"   能带 {band_idx+1}: {behavior['description']}")
        
        # 2. 找出所有真正穿过费米能级的能带
        print("\n2. 查找真正穿过费米能级的能带:")
        crossing_bands = []
        for band_idx, behavior in enumerate(band_behavior):
            if behavior['crosses_fermi']:
                crossing_info = {
                    'band_idx': band_idx,
                    'crossing_points': behavior['crossing_points'],
                    'is_same_point': behavior['is_same_point']
                }
                crossing_bands.append(crossing_info)
                print(f"   能带 {band_idx+1} 真正穿过费米能级")
                for cp in behavior['crossing_points']:
                    symbol_text = f" ({cp['symbol']})" if cp['symbol'] else ""
                    print(f"     - 在k点 {cp['k_idx']}{symbol_text}, 路径坐标: {cp['k_path']:.4f}")
        
        print(f"\n   真正穿过费米能级的能带数: {len(crossing_bands)}")
        
        # 特殊处理：如果从AGR文件中读取了费米能级，调整能带数据
        if self.efermi_original != 0.0:
            print(f"注意: 从AGR文件中读取了费米能级 {self.efermi_original} eV")
            print("检查价带顶和导带底是否基于正确的费米能级")
        
        # 3. 根据情况分类处理
        if len(crossing_bands) == 0:
            # 情况A: 没有能带真正穿过费米能级
            print("\n3. 情况A: 没有能带真正穿过费米能级")
            self.band_gap_info = self._analyze_case_no_crossing(band_behavior)
        
        elif len(crossing_bands) == 1:
            # 情况B: 只有一条能带穿过费米能级
            print("\n3. 情况B: 只有一条能带穿过费米能级")
            self.band_gap_info = self._analyze_case_single_crossing(
                crossing_bands[0], band_behavior, interactive_mode
            )
        
        else:
            # 情况C: 多条能带穿过费米能级
            print("\n3. 情况C: 多条能带穿过费米能级")
            # 检查是否所有穿越点都接近费米能级
            all_close = True
            for cb in crossing_bands:
                for cp in cb['crossing_points']:
                    # 获取该k点的实际能量
                    energy = self.band_data[cb['band_idx']][cp['k_idx']]
                    # 检查是否接近费米能级
                    if abs(energy - self.fermi_energy) > 0.1:  # 100 meV阈值
                        all_close = False
                        break
                if not all_close:
                    break
            
            if all_close:
                print("   所有穿越点都接近费米能级，可能是金属")
                self.band_gap_info = self._analyze_case_multiple_crossing(
                    crossing_bands, band_behavior, interactive_mode
                )
            else:
                print("   穿越点远离费米能级，重新检查带隙")
                # 尝试重新分析，不考虑穿越点
                self.band_gap_info = self._analyze_case_no_crossing(band_behavior)
        
        # 输出分析结果
        print("\n4. 带隙分析结果:")
        if self.band_gap_info:
            self._display_band_gap_results(self.band_gap_info)
        
        return self.band_gap_info
    
    def _analyze_single_band(self, band_idx, crossing_threshold=1e-6):
        """分析单条能带的行为"""
        energies = self.band_data[band_idx]
        
        # 检查能带是否真正穿过费米能级（从负到正或正到负）
        crossing_points = []
        for i in range(len(energies) - 1):
            e1 = energies[i]
            e2 = energies[i + 1]
            
            # 检查是否跨越费米能级（当前费米能级为参考点）
            if (e1 - self.fermi_energy) * (e2 - self.fermi_energy) < 0:  # 异号，说明跨越了费米能级
                # 线性插值找到精确的穿越点
                k1 = self.k_path[1][i]  # 归一化的路径坐标
                k2 = self.k_path[1][i + 1]
                # 使用当前费米能级作为参考点
                e1_rel = e1 - self.fermi_energy
                e2_rel = e2 - self.fermi_energy
                fraction = abs(e1_rel) / (abs(e1_rel) + abs(e2_rel))
                k_cross = k1 + fraction * (k2 - k1)
                
                # 获取最近k点的坐标信息
                nearest_idx = i if abs(e1_rel) < abs(e2_rel) else i + 1
                k_coords = self.k_points[nearest_idx]
                symbol = self.get_k_point_symbol(k_coords)
                
                crossing_points.append({
                    'k_idx': nearest_idx,
                    'k_path': k_cross,
                    'k_coords': k_coords,
                    'symbol': symbol,
                    'sign_change': 'negative_to_positive' if e1_rel < 0 else 'positive_to_negative'
                })
        
        # 检查是否有能量刚好在费米能级上的点
        zero_points = []
        for i, energy in enumerate(energies):
            if abs(energy - self.fermi_energy) < crossing_threshold:
                k_coords = self.k_points[i]
                symbol = self.get_k_point_symbol(k_coords)
                zero_points.append({
                    'k_idx': i,
                    'k_path': self.k_path[1][i],
                    'k_coords': k_coords,
                    'symbol': symbol,
                    'energy': energy
                })
        
        # 判断所有穿越点是否是同一个等价高对称点
        is_same_point = False
        if crossing_points:
            first_symbol = crossing_points[0]['symbol']
            if first_symbol:
                same_count = sum(1 for cp in crossing_points if cp['symbol'] == first_symbol)
                is_same_point = (same_count == len(crossing_points))
        
        # 分析能带的整体行为
        min_energy = min(energies)
        max_energy = max(energies)
        
        # 判断能带整体在费米能级的哪一侧
        if max_energy <= self.fermi_energy:
            overall_position = 'below_fermi'
        elif min_energy >= self.fermi_energy:
            overall_position = 'above_fermi'
        else:
            overall_position = 'crosses_fermi'
        
        # 生成描述
        if crossing_points:
            if is_same_point and crossing_points[0]['symbol']:
                description = f"真正穿过费米能级 ({len(crossing_points)}次，都在{crossing_points[0]['symbol']}点)"
            else:
                description = f"真正穿过费米能级 ({len(crossing_points)}次)"
        elif zero_points:
            if zero_points and zero_points[0]['symbol']:
                description = f"刚好在费米能级上 ({len(zero_points)}个点，如{zero_points[0]['symbol']}点)"
            else:
                description = f"刚好在费米能级上 ({len(zero_points)}个点)"
        elif overall_position == 'below_fermi':
            description = "完全在费米能级以下"
        elif overall_position == 'above_fermi':
            description = "完全在费米能级以上"
        else:
            description = "部分在费米能级以上，部分以下但未穿越"
        
        return {
            'band_idx': band_idx,
            'crosses_fermi': len(crossing_points) > 0,
            'crossing_points': crossing_points,
            'zero_points': zero_points,
            'is_same_point': is_same_point,
            'overall_position': overall_position,
            'min_energy': min_energy,
            'max_energy': max_energy,
            'description': description,
            'energies': energies
        }
    
    def _analyze_case_no_crossing(self, band_behavior):
        """情况A: 没有能带真正穿过费米能级"""
        print("   分析: 没有能带真正穿过费米能级，寻找价带顶和导带底")
        
        # 寻找价带顶（所有低于费米能级的能量中的最大值）
        valence_candidates = []
        for behavior in band_behavior:
            if behavior['max_energy'] <= self.fermi_energy:  # 整个能带都在费米能级以下
                # 找到这个能带的最大能量点
                energies = behavior['energies']
                max_val = max(energies)
                max_idx = list(energies).index(max_val)
                
                k_coords = self.k_points[max_idx]
                symbol = self.get_k_point_symbol(k_coords)
                
                valence_candidates.append({
                    'energy': max_val,
                    'band_idx': behavior['band_idx'],
                    'k_idx': max_idx,
                    'k_path': self.k_path[1][max_idx],  # 归一化的路径坐标
                    'k_coords': k_coords,
                    'symbol': symbol,
                    'description': f"能带{behavior['band_idx']+1}的最大值"
                })
            elif behavior['min_energy'] < self.fermi_energy:  # 部分在费米能级以下
                # 找到低于费米能级的能量中的最大值
                energies = np.array(behavior['energies'])
                below_fermi_indices = np.where(energies <= self.fermi_energy)[0]
                if len(below_fermi_indices) > 0:
                    below_fermi_energies = energies[below_fermi_indices]
                    max_below_idx = below_fermi_indices[np.argmax(below_fermi_energies)]
                    
                    k_coords = self.k_points[max_below_idx]
                    symbol = self.get_k_point_symbol(k_coords)
                    
                    valence_candidates.append({
                        'energy': energies[max_below_idx],
                        'band_idx': behavior['band_idx'],
                        'k_idx': max_below_idx,
                        'k_path': self.k_path[1][max_below_idx],
                        'k_coords': k_coords,
                        'symbol': symbol,
                        'description': f"能带{behavior['band_idx']+1}的低于费米能级部分最大值"
                    })
        
        # 寻找导带底（所有高于费米能级的能量中的最小值）
        conduction_candidates = []
        for behavior in band_behavior:
            if behavior['min_energy'] >= self.fermi_energy:  # 整个能带都在费米能级以上
                # 找到这个能带的最小能量点
                energies = behavior['energies']
                min_val = min(energies)
                min_idx = list(energies).index(min_val)
                
                k_coords = self.k_points[min_idx]
                symbol = self.get_k_point_symbol(k_coords)
                
                conduction_candidates.append({
                    'energy': min_val,
                    'band_idx': behavior['band_idx'],
                    'k_idx': min_idx,
                    'k_path': self.k_path[1][min_idx],
                    'k_coords': k_coords,
                    'symbol': symbol,
                    'description': f"能带{behavior['band_idx']+1}的最小值"
                })
            elif behavior['max_energy'] > self.fermi_energy:  # 部分在费米能级以上
                # 找到高于费米能级的能量中的最小值
                energies = np.array(behavior['energies'])
                above_fermi_indices = np.where(energies >= self.fermi_energy)[0]
                if len(above_fermi_indices) > 0:
                    above_fermi_energies = energies[above_fermi_indices]
                    min_above_idx = above_fermi_indices[np.argmin(above_fermi_energies)]
                    
                    k_coords = self.k_points[min_above_idx]
                    symbol = self.get_k_point_symbol(k_coords)
                    
                    conduction_candidates.append({
                        'energy': energies[min_above_idx],
                        'band_idx': behavior['band_idx'],
                        'k_idx': min_above_idx,
                        'k_path': self.k_path[1][min_above_idx],
                        'k_coords': k_coords,
                        'symbol': symbol,
                        'description': f"能带{behavior['band_idx']+1}的高于费米能级部分最小值"
                    })
        
        # 选择最佳的价带顶和导带底
        if valence_candidates:
            valence_top = max(valence_candidates, key=lambda x: x['energy'])
        else:
            print("警告: 未找到价带顶")
            valence_top = None
        
        if conduction_candidates:
            conduction_bottom = min(conduction_candidates, key=lambda x: x['energy'])
        else:
            print("警告: 未找到导带底")
            conduction_bottom = None
        
        # 计算带隙
        if valence_top and conduction_bottom:
            band_gap = conduction_bottom['energy'] - valence_top['energy']
            is_direct = (valence_top['k_idx'] == conduction_bottom['k_idx'])
            
            print(f"价带顶: {valence_top['energy']:.6f} eV (能带 {valence_top['band_idx']+1})")
            print(f"导带底: {conduction_bottom['energy']:.6f} eV (能带 {conduction_bottom['band_idx']+1})")
            print(f"带隙: {band_gap:.6f} eV")
            print(f"带隙类型: {'直接' if is_direct else '间接'}带隙")
            
            return {
                'case': 'semiconductor',
                'valence_top': valence_top,
                'conduction_bottom': conduction_bottom,
                'band_gap': band_gap,
                'is_direct': is_direct,
                'material_type': self._classify_material(band_gap),
                'note': '标准半导体：价带顶在费米能级以下，导带底在费米能级以上'
            }
        else:
            print("错误: 无法确定价带顶和导带底")
            return {
                'case': 'unknown',
                'note': '无法确定价带顶和导带底'
            }
    
    def _analyze_case_single_crossing(self, crossing_band, band_behavior, interactive_mode):
        """情况B: 只有一条能带穿过费米能级"""
        band_idx = crossing_band['band_idx']
        behavior = band_behavior[band_idx]
        
        print(f"   分析能带 {band_idx+1} 的穿越行为:")
        
        # 检查穿越点是否都是同一个等价高对称点
        if crossing_band['is_same_point'] and behavior['crossing_points'] and behavior['crossing_points'][0]['symbol']:
            symbol = behavior['crossing_points'][0]['symbol']
            print(f"    所有穿越点都在同一个高对称点: {symbol}")
            
            # 检查能带整体行为
            if behavior['overall_position'] == 'crosses_fermi':
                print(f"    能带在{symbol}点穿越费米能级")
                
                # 检查能带在其他点的行为
                other_energies = []
                for i, energy in enumerate(behavior['energies']):
                    is_crossing_point = any(cp['k_idx'] == i for cp in behavior['crossing_points'])
                    if not is_crossing_point:
                        other_energies.append(energy)
                
                if other_energies:
                    avg_other = np.mean(other_energies)
                    print(f"    其他点平均能量: {avg_other:.4f} eV")
                    
                    if avg_other < self.fermi_energy - 0.01:  # 大部分点在费米能级以下
                        print(f"    结论: 这是价带顶刚好在费米能级上")
                        
                        # 寻找其他能带的导带底
                        conduction_candidates = []
                        for b_idx, b_behavior in enumerate(band_behavior):
                            if b_idx == band_idx:
                                continue
                            
                            # 找到高于费米能级的能量中的最小值
                            energies = np.array(b_behavior['energies'])
                            positive_indices = np.where(energies > self.fermi_energy + 0.01)[0]
                            if len(positive_indices) > 0:
                                positive_energies = energies[positive_indices]
                                min_pos_idx = positive_indices[np.argmin(positive_energies)]
                                
                                k_coords = self.k_points[min_pos_idx]
                                symbol_cb = self.get_k_point_symbol(k_coords)
                                
                                conduction_candidates.append({
                                    'energy': energies[min_pos_idx],
                                    'band_idx': b_idx,
                                    'k_idx': min_pos_idx,
                                    'k_path': self.k_path[1][min_pos_idx],
                                    'k_coords': k_coords,
                                    'symbol': symbol_cb,
                                    'description': f"能带{b_idx+1}的最小正能量"
                                })
                        
                        if conduction_candidates:
                            conduction_bottom = min(conduction_candidates, key=lambda x: x['energy'])
                            band_gap = conduction_bottom['energy'] - self.fermi_energy  # 价带顶在当前费米能级
                            
                            # 创建假的价带顶信息
                            valence_top = {
                                'energy': self.fermi_energy,
                                'band_idx': band_idx,
                                'k_idx': behavior['crossing_points'][0]['k_idx'],
                                'k_path': behavior['crossing_points'][0]['k_path'],
                                'k_coords': behavior['crossing_points'][0]['k_coords'],
                                'symbol': symbol,
                                'description': f"能带{band_idx+1}在{symbol}点刚好在费米能级"
                            }
                            
                            is_direct = (valence_top['k_idx'] == conduction_bottom['k_idx'])
                            
                            return {
                                'case': 'valence_top_at_fermi',
                                'valence_top': valence_top,
                                'conduction_bottom': conduction_bottom,
                                'band_gap': band_gap,
                                'is_direct': is_direct,
                                'material_type': self._classify_material(band_gap),
                                'note': f'能带{band_idx+1}在{symbol}点刚好在费米能级上作为价带顶'
                            }
                    
                    elif avg_other > self.fermi_energy + 0.01:  # 大部分点在费米能级以上
                        print(f"    结论: 这是导带底刚好在费米能级上")
                        
                        # 寻找其他能带的价带顶
                        valence_candidates = []
                        for b_idx, b_behavior in enumerate(band_behavior):
                            if b_idx == band_idx:
                                continue
                            
                            # 找到低于费米能级的能量中的最大值
                            energies = np.array(b_behavior['energies'])
                            negative_indices = np.where(energies < self.fermi_energy - 0.01)[0]
                            if len(negative_indices) > 0:
                                negative_energies = energies[negative_indices]
                                max_neg_idx = negative_indices[np.argmax(negative_energies)]
                                
                                k_coords = self.k_points[max_neg_idx]
                                symbol_vb = self.get_k_point_symbol(k_coords)
                                
                                valence_candidates.append({
                                    'energy': energies[max_neg_idx],
                                    'band_idx': b_idx,
                                    'k_idx': max_neg_idx,
                                    'k_path': self.k_path[1][max_neg_idx],
                                    'k_coords': k_coords,
                                    'symbol': symbol_vb,
                                    'description': f"能带{b_idx+1}的最大负能量"
                                })
                        
                        if valence_candidates:
                            valence_top = max(valence_candidates, key=lambda x: x['energy'])
                            band_gap = self.fermi_energy - valence_top['energy']  # 导带底在当前费米能级
                            
                            # 创建假的导带底信息
                            conduction_bottom = {
                                'energy': self.fermi_energy,
                                'band_idx': band_idx,
                                'k_idx': behavior['crossing_points'][0]['k_idx'],
                                'k_path': behavior['crossing_points'][0]['k_path'],
                                'k_coords': behavior['crossing_points'][0]['k_coords'],
                                'symbol': symbol,
                                'description': f"能带{band_idx+1}在{symbol}点刚好在费米能级"
                            }
                            
                            is_direct = (valence_top['k_idx'] == conduction_bottom['k_idx'])
                            
                            return {
                                'case': 'conduction_bottom_at_fermi',
                                'valence_top': valence_top,
                                'conduction_bottom': conduction_bottom,
                                'band_gap': band_gap,
                                'is_direct': is_direct,
                                'material_type': self._classify_material(band_gap),
                                'note': f'能带{band_idx+1}在{symbol}点刚好在费米能级上作为导带底'
                            }
        
        # 如果穿越点在不同位置，可能是金属
        print(f"    能带在不同位置穿越费米能级，可能是金属")
        
        # 检查与其他能带的交叉
        intersecting_bands = self._check_band_intersections(band_idx)
        
        if intersecting_bands:
            print(f"    与能带 {[b+1 for b in intersecting_bands]} 有交叉")
            
            if interactive_mode:
                response = input("是否检查带隙可能性？(y/n): ").strip().lower()
                if response == 'y':
                    return self._check_gap_possibility(band_idx, behavior, band_behavior)
        
        # 默认判断为金属
        return {
            'case': 'metal',
            'crossing_band': band_idx,
            'crossing_points': crossing_band['crossing_points'],
            'note': f'能带{band_idx+1}在不同位置穿越费米能级，可能为金属'
        }
    
    def _analyze_case_multiple_crossing(self, crossing_bands, band_behavior, interactive_mode):
        """情况C: 多条能带穿过费米能级"""
        print("   多条能带穿过费米能级，可能为金属")
        
        if interactive_mode:
            print("   需要用户交互分析...")
            # 简化处理：询问是否分析特定能带
            response = input("是否分析特定能带的带隙可能性？(y/n): ").strip().lower()
            if response == 'y':
                band_num = int(input(f"选择要分析的能带编号(1-{self.num_bands}): ")) - 1
                if 0 <= band_num < self.num_bands:
                    behavior = band_behavior[band_num]
                    if behavior['crosses_fermi']:
                        return self._check_gap_possibility(band_num, behavior, band_behavior)
        
        # 默认返回金属特性
        return {
            'case': 'metal_multiple_bands',
            'crossing_bands': [cb['band_idx'] for cb in crossing_bands],
            'note': f'多条能带{[cb["band_idx"]+1 for cb in crossing_bands]}穿过费米能级，可能为金属'
        }
    
    def _check_band_intersections(self, band_idx, threshold=0.1):
        """检查能带是否与其他能带相交"""
        intersecting_bands = []
        
        for other_idx in range(self.num_bands):
            if other_idx == band_idx:
                continue
            
            # 检查每个k点的能量差
            for k_idx in range(self.num_kpoints):
                energy1 = self.band_data[band_idx][k_idx]
                energy2 = self.band_data[other_idx][k_idx]
                
                if abs(energy1 - energy2) < threshold:
                    if other_idx not in intersecting_bands:
                        intersecting_bands.append(other_idx)
                    break
        
        return intersecting_bands
    
    def _check_gap_possibility(self, band_idx, behavior, band_behavior):
        """检查带隙可能性"""
        # 寻找其他能带的导带底
        conduction_candidates = []
        for b_idx, b_behavior in enumerate(band_behavior):
            if b_idx == band_idx:
                continue
            
            # 找到高于费米能级的能量中的最小值
            energies = np.array(b_behavior['energies'])
            positive_indices = np.where(energies > self.fermi_energy + 0.01)[0]
            if len(positive_indices) > 0:
                positive_energies = energies[positive_indices]
                min_pos_idx = positive_indices[np.argmin(positive_energies)]
                
                k_coords = self.k_points[min_pos_idx]
                symbol = self.get_k_point_symbol(k_coords)
                
                conduction_candidates.append({
                    'energy': energies[min_pos_idx],
                    'band_idx': b_idx,
                    'k_idx': min_pos_idx,
                    'k_path': self.k_path[1][min_pos_idx],
                    'k_coords': k_coords,
                    'symbol': symbol
                })
        
        if conduction_candidates:
            conduction_bottom = min(conduction_candidates, key=lambda x: x['energy'])
            
            # 假设当前能带在穿越点作为价带顶
            valence_top = {
                'energy': self.fermi_energy,
                'band_idx': band_idx,
                'k_idx': behavior['crossing_points'][0]['k_idx'],
                'k_path': behavior['crossing_points'][0]['k_path'],
                'k_coords': behavior['crossing_points'][0]['k_coords'],
                'symbol': behavior['crossing_points'][0]['symbol'],
            }
            
            band_gap = conduction_bottom['energy'] - self.fermi_energy
            is_direct = (valence_top['k_idx'] == conduction_bottom['k_idx'])
            
            return {
                'case': 'possible_indirect_gap',
                'valence_top': valence_top,
                'conduction_bottom': conduction_bottom,
                'band_gap': band_gap,
                'is_direct': is_direct,
                'material_type': self._classify_material(band_gap),
                'note': f'假设能带{band_idx+1}在穿越点作为价带顶'
            }
        
        return {
            'case': 'metal_no_conduction',
            'note': '未找到其他能带的导带底，可能为金属'
        }
    
    def _classify_material(self, band_gap):
        """根据带隙大小分类材料"""
        if band_gap < 0.01:
            return "金属/半金属"
        elif band_gap < 1.5:
            return "窄带隙半导体"
        elif band_gap < 3.0:
            return "半导体"
        else:
            return "宽带隙半导体/绝缘体"
    
    def _display_band_gap_results(self, band_gap_info):
        """显示带隙分析结果"""
        case = band_gap_info.get('case', 'unknown')
        
        if case == 'semiconductor':
            v = band_gap_info['valence_top']
            c = band_gap_info['conduction_bottom']
            
            print(f"   结果: 半导体材料")
            print(f"   价带顶: {v['energy']:.6f} eV (能带 {v['band_idx']+1})")
            if v['symbol']:
                print(f"     位置: {v['symbol']}点")
            else:
                print(f"     位置: k点 {v['k_idx']} (路径坐标: {v['k_path']:.4f})")
            
            print(f"   导带底: {c['energy']:.6f} eV (能带 {c['band_idx']+1})")
            if c['symbol']:
                print(f"     位置: {c['symbol']}点")
            else:
                print(f"     位置: k点 {c['k_idx']} (路径坐标: {c['k_path']:.4f})")
            
            print(f"   带隙: {band_gap_info['band_gap']:.6f} eV")
            print(f"   带隙类型: {'直接' if band_gap_info['is_direct'] else '间接'}带隙")
            print(f"   材料类型: {band_gap_info['material_type']}")
        
        elif case in ['valence_top_at_fermi', 'conduction_bottom_at_fermi']:
            v = band_gap_info['valence_top']
            c = band_gap_info['conduction_bottom']
            
            if case == 'valence_top_at_fermi':
                print(f"   结果: 价带顶刚好在费米能级上")
            else:
                print(f"   结果: 导带底刚好在费米能级上")
            
            print(f"   说明: {band_gap_info.get('note', '')}")
            print(f"   价带顶: {v['energy']:.6f} eV (能带 {v['band_idx']+1})")
            print(f"   导带底: {c['energy']:.6f} eV (能带 {c['band_idx']+1})")
            print(f"   带隙: {band_gap_info['band_gap']:.6f} eV")
            print(f"   材料类型: {band_gap_info['material_type']}")
        
        elif case == 'metal':
            print(f"   结果: 金属性材料")
            print(f"   说明: {band_gap_info.get('note', '')}")
            if 'crossing_points' in band_gap_info:
                print(f"   穿越费米能级的位置:")
                for cp in band_gap_info['crossing_points']:
                    if cp['symbol']:
                        print(f"     {cp['symbol']}点 (路径坐标: {cp['k_path']:.4f})")
        
        elif case == 'metal_multiple_bands':
            print(f"   结果: 金属性材料（多条能带穿越费米能级）")
            print(f"   说明: {band_gap_info.get('note', '')}")
        
        elif case == 'possible_indirect_gap':
            print(f"   结果: 可能的间接带隙半导体")
            print(f"   说明: {band_gap_info.get('note', '')}")
            print(f"   带隙: {band_gap_info['band_gap']:.6f} eV")
            print(f"   材料类型: {band_gap_info['material_type']}")
        
        else:
            print(f"   结果: {band_gap_info.get('note', '未知情况')}")
    
    def organize_band_data(self, output_csv=None):
        """整理能带数据并保存为CSV格式"""
        if not self.band_data or not self.k_points:
            print("错误: 数据未正确解析")
            return None
        
        if self.k_path is None:
            print("错误: 未计算k路径坐标")
            return None
        
        # 准备输出数据
        output_data = []
        
        # 表头
        header = ["k_index", "k_path", "k_path_norm"]
        for i in range(self.num_bands):
            header.append(f"band_{i+1}")
        
        output_data.append(header)
        
        # 数据行
        for i in range(self.num_kpoints):
            row = [i, f"{self.k_path[0][i]:.6f}", f"{self.k_path[1][i]:.6f}"]
            for band_idx in range(self.num_bands):
                if i < len(self.band_data[band_idx]):
                    row.append(f"{self.band_data[band_idx][i]:.9f}")
                else:
                    row.append("NaN")
            output_data.append(row)
        
        # 保存为CSV文件
        if output_csv:
            try:
                with open(output_csv, 'w', encoding='utf-8') as f:
                    for row in output_data:
                        f.write(','.join(map(str, row)) + '\n')
                print(f"数据已保存到: {output_csv}")
            except Exception as e:
                print(f"错误: 无法保存CSV文件 - {e}")
        
        return output_data
    
    def get_high_symmetry_points_for_plot(self):
        """获取用于绘图的高对称点信息"""
        if self.k_path is None:
            return [], []
        
        # 首先尝试从输入文件中找到的高对称点路径
        high_sym_indices = self.find_high_symmetry_indices()
        
        if high_sym_indices:
            tick_positions = []
            tick_labels = []
            
            for label, idx in sorted(high_sym_indices.items(), key=lambda x: x[1]):
                if 0 <= idx < len(self.k_path[1]):  # 使用归一化的路径坐标
                    tick_positions.append(self.k_path[1][idx])
                    tick_labels.append(label)
            
            # 确保包含起点和终点
            if 0 not in high_sym_indices.values():
                tick_positions.insert(0, self.k_path[1][0])
                tick_labels.insert(0, "Start")
            
            if self.num_kpoints-1 not in high_sym_indices.values():
                tick_positions.append(self.k_path[1][-1])
                tick_labels.append("End")
            
            return tick_positions, tick_labels
        
        # 如果没有从输入文件找到，使用AGR文件中的信息
        elif self.high_sym_points:
            tick_positions = []
            tick_labels = []
            
            for label, coord in self.high_sym_points.items():
                idx = int(coord)
                if 0 <= idx < len(self.k_path[1]):
                    tick_positions.append(self.k_path[1][idx])
                    tick_labels.append(label)
            
            return tick_positions, tick_labels
        
        # 都没有，使用默认的重要点
        else:
            # 使用一些常见的k点索引作为高对称点
            default_points = {
                0: "Γ",
                57: "X",
                85: "W"
            }
            
            tick_positions = []
            tick_labels = []
            
            for idx, label in default_points.items():
                if idx < len(self.k_path[1]):
                    tick_positions.append(self.k_path[1][idx])
                    tick_labels.append(label)
            
            return tick_positions, tick_labels
    
    def adjust_fermi_level(self):
        """
        调整费米能级的交互功能
        """
        print("\n" + "="*70)
        print("费米能级调整功能")
        print("="*70)
        print(f"当前费米能级: {self.fermi_energy:.6f} eV")
        print(f"原始费米能级: {self.efermi_original:.6f} eV")
        
        while True:
            print("\n请选择操作:")
            print("1. 向上调整费米能级")
            print("2. 向下调整费米能级")
            print("3. 直接设置费米能级")
            print("4. 重置为原始费米能级")
            print("5. 完成调整")
            
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                try:
                    delta = float(input("请输入向上调整的能量值 (eV): ").strip())
                    self.fermi_energy += delta
                    print(f"费米能级已向上调整 {delta:.6f} eV，当前值: {self.fermi_energy:.6f} eV")
                except ValueError:
                    print("输入无效，请输入数字")
            
            elif choice == '2':
                try:
                    delta = float(input("请输入向下调整的能量值 (eV): ").strip())
                    self.fermi_energy -= delta
                    print(f"费米能级已向下调整 {delta:.6f} eV，当前值: {self.fermi_energy:.6f} eV")
                except ValueError:
                    print("输入无效，请输入数字")
            
            elif choice == '3':
                try:
                    new_fermi = float(input("请输入新的费米能级值 (eV): ").strip())
                    self.fermi_energy = new_fermi
                    print(f"费米能级已设置为: {self.fermi_energy:.6f} eV")
                except ValueError:
                    print("输入无效，请输入数字")
            
            elif choice == '4':
                self.fermi_energy = 0.0  # 重置为原始参考点
                print(f"费米能级已重置为: {self.fermi_energy:.6f} eV")
            
            elif choice == '5':
                print("\n费米能级调整完成!")
                print(f"最终费米能级: {self.fermi_energy:.6f} eV")
                break
            
            else:
                print("输入无效，请输入1-5之间的数字")
            
            # 重新分析带隙
            print("\n重新分析带隙...")
            self.analyze_band_gap()
            
        return self.fermi_energy
    
    def plot_band_structure(self, output_image=None, dpi=300, show_plot=True):
        """绘制能带结构图"""
        if not self.band_data or not self.k_points:
            print("错误: 数据未正确解析，无法绘图")
            return
        
        if self.k_path is None:
            print("错误: 未计算k路径坐标")
            return
        
        # 创建图形
        plt.figure(figsize=(14, 8))
        
        # 定义颜色列表（为每条能带分配不同颜色）
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, self.num_bands)))
        
        # 绘制每条能带
        for band_idx in range(self.num_bands):
            if len(self.band_data[band_idx]) == len(self.k_path[1]):
                plt.plot(self.k_path[1], self.band_data[band_idx], 
                        color=colors[band_idx % len(colors)], 
                        linewidth=1.5, 
                        alpha=0.8,
                        label=f'Band {band_idx+1}')
        
        # 获取高对称点信息
        tick_positions, tick_labels = self.get_high_symmetry_points_for_plot()
        
        # 添加高对称点垂直线
        for pos in tick_positions:
            plt.axvline(x=pos, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # 设置x轴范围
        plt.xlim(min(self.k_path[1]), max(self.k_path[1]))
        
        # 设置x轴刻度 - 显示所有重要高对称点
        if tick_positions and tick_labels:
            plt.xticks(tick_positions, tick_labels, fontsize=12)
        
        # 添加费米能级线（灰色虚线，不显眼）
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # 设置图形属性
        plt.xlabel('Wave Vector', fontsize=14)
        plt.ylabel('Energy (eV)', fontsize=14)
        
        # 根据是否有原始费米能级信息设置标题
        if self.efermi_original:
            plt.title(f'Band Structure (Fermi level shifted from {self.efermi_original:.3f} eV to 0 eV)', 
                     fontsize=16, fontweight='bold')
        else:
            plt.title('Band Structure', fontsize=16, fontweight='bold')
        
        # 添加网格
        plt.grid(True, alpha=0.2, linestyle=':')
        
        # 自动设置y轴范围（考虑所有能带）
        all_energies = []
        for band in self.band_data:
            all_energies.extend(band)
        
        y_min, y_max = min(all_energies), max(all_energies)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # 在图上标注带隙信息
        if self.band_gap_info:
            self._annotate_band_gap_on_plot()
        
        # 添加图例（如果带数不太多）
        if self.num_bands <= 20:
            plt.legend(loc='upper right', fontsize=8, ncol=2)
        
        # 添加文本说明
        info_text = f'Bands: {self.num_bands}, k-points: {self.num_kpoints}'
        if self.lattice_constant:
            info_text = f'Lattice constant: {self.lattice_constant:.4f} Å\n' + info_text
        
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        if output_image:
            try:
                plt.savefig(output_image, dpi=dpi, bbox_inches='tight')
                print(f"能带图已保存到: {output_image}")
            except Exception as e:
                print(f"错误: 无法保存图像 - {e}")
        
        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # 输出高对称点信息
        print("\n能带图高对称点标记:")
        print("-" * 60)
        for pos, label in zip(tick_positions, tick_labels):
            # 找到对应的k点索引
            k_idx = None
            for i, kp in enumerate(self.k_path[1]):
                if abs(kp - pos) < 1e-6:
                    k_idx = i
                    break
            
            if k_idx is not None and k_idx < len(self.k_points):
                k_coord = self.k_points[k_idx]
                print(f"{label}: k点索引 = {k_idx}, 路径坐标 = {pos:.6f}, 分数坐标 = [{k_coord[0]:.4f}, {k_coord[1]:.4f}, {k_coord[2]:.4f}]")
    
    def _annotate_band_gap_on_plot(self):
        """在图上标注带隙信息"""
        if not self.band_gap_info:
            return
        
        case = self.band_gap_info.get('case', 'unknown')
        
        # 准备标注文本
        annotation_text = ""
        
        if case == 'semiconductor':
            v = self.band_gap_info['valence_top']
            c = self.band_gap_info['conduction_bottom']
            
            annotation_text = f"Band Gap: {self.band_gap_info['band_gap']:.3f} eV\n"
            annotation_text += f"Type: {'Direct' if self.band_gap_info['is_direct'] else 'Indirect'}\n"
            
            v_symbol = v['symbol'] if v['symbol'] else f"k={v['k_idx']}"
            c_symbol = c['symbol'] if c['symbol'] else f"k={c['k_idx']}"
            
            annotation_text += f"VBM: {v_symbol} (Band {v['band_idx']+1})\n"
            annotation_text += f"CBM: {c_symbol} (Band {c['band_idx']+1})"
            
            # 在价带顶和导带底处添加标记
            plt.plot(v['k_path'], v['energy'], 'ro', markersize=8, label='VBM')
            plt.plot(c['k_path'], c['energy'], 'go', markersize=8, label='CBM')
        
        elif case in ['valence_top_at_fermi', 'conduction_bottom_at_fermi']:
            v = self.band_gap_info['valence_top']
            c = self.band_gap_info['conduction_bottom']
            
            annotation_text = f"Band Gap: {self.band_gap_info['band_gap']:.3f} eV\n"
            
            if case == 'valence_top_at_fermi':
                annotation_text += "VBM at Fermi level\n"
            else:
                annotation_text += "CBM at Fermi level\n"
            
            v_symbol = v['symbol'] if v['symbol'] else f"k={v['k_idx']}"
            c_symbol = c['symbol'] if c['symbol'] else f"k={c['k_idx']}"
            
            annotation_text += f"VBM: {v_symbol}\n"
            annotation_text += f"CBM: {c_symbol}"
            
            # 添加标记
            if v['energy'] != 0:
                plt.plot(v['k_path'], v['energy'], 'ro', markersize=8, label='VBM')
            if c['energy'] != 0:
                plt.plot(c['k_path'], c['energy'], 'go', markersize=8, label='CBM')
        
        elif case == 'metal':
            annotation_text = "Metallic behavior\n"
            if 'crossing_points' in self.band_gap_info:
                annotation_text += f"Band {self.band_gap_info['crossing_band']+1} crosses Fermi level"
            
            # 标记穿越点
            if 'crossing_points' in self.band_gap_info:
                for cp in self.band_gap_info['crossing_points']:
                    plt.plot(cp['k_path'], 0, 'mo', markersize=6, alpha=0.7)
        
        elif case == 'metal_multiple_bands':
            annotation_text = "Metallic behavior\n"
            annotation_text += f"{len(self.band_gap_info['crossing_bands'])} bands cross Fermi level"
        
        elif case == 'possible_indirect_gap':
            annotation_text = f"Possible indirect gap: {self.band_gap_info['band_gap']:.3f} eV\n"
            annotation_text += f"Assumed VBM at Fermi level"
        
        # 添加标注文本框
        if annotation_text:
            plt.text(0.98, 0.02, annotation_text,
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def generate_summary_report(self, output_file=None):
        """生成分析报告"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ABINIT能带数据分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"分析文件: {os.path.basename(self.agr_file)}")
        report_lines.append(f"文件路径: {os.path.dirname(os.path.abspath(self.agr_file))}")
        if self.abi_file:
            report_lines.append(f"输入文件: {os.path.basename(self.abi_file)}")
        report_lines.append(f"分析时间: {np.datetime64('now')}")
        report_lines.append("")
        
        # 晶体结构信息
        report_lines.append("1. 晶体结构信息")
        report_lines.append("-" * 40)
        if self.lattice_constant:
            report_lines.append(f"晶格常数: {self.lattice_constant} Å")
        if self.rprim is not None:
            report_lines.append(f"原胞基矢:")
            report_lines.append(f"  a₁ = [{self.rprim[0,0]:.6f}, {self.rprim[0,1]:.6f}, {self.rprim[0,2]:.6f}]")
            report_lines.append(f"  a₂ = [{self.rprim[1,0]:.6f}, {self.rprim[1,1]:.6f}, {self.rprim[1,2]:.6f}]")
            report_lines.append(f"  a₃ = [{self.rprim[2,0]:.6f}, {self.rprim[2,1]:.6f}, {self.rprim[2,2]:.6f}]")
        report_lines.append("")
        
        # 计算信息
        report_lines.append("2. 计算参数")
        report_lines.append("-" * 40)
        report_lines.append(f"能带数量: {self.num_bands}")
        report_lines.append(f"k点数量: {self.num_kpoints}")
        if self.efermi_original:
            report_lines.append(f"原始费米能级: {self.efermi_original:.6f} eV")
        report_lines.append(f"当前费米能级: {self.fermi_energy} eV")
        report_lines.append("")
        
        # 高对称点信息
        report_lines.append("3. 高对称点信息")
        report_lines.append("-" * 40)
        
        tick_positions, tick_labels = self.get_high_symmetry_points_for_plot()
        
        for pos, label in zip(tick_positions, tick_labels):
            # 找到对应的k点索引
            k_idx = None
            for i, kp in enumerate(self.k_path[1]):
                if abs(kp - pos) < 1e-6:
                    k_idx = i
                    break
            
            if k_idx is not None and k_idx < len(self.k_points):
                k_coord = self.k_points[k_idx]
                report_lines.append(f"{label:<15} k点索引: {k_idx:<4} 归一化坐标: {pos:.6f} 分数坐标: [{k_coord[0]:.4f}, {k_coord[1]:.4f}, {k_coord[2]:.4f}]")
        
        report_lines.append("")
        
        # 带隙分析
        report_lines.append("4. 带隙分析")
        report_lines.append("-" * 40)
        
        if self.band_gap_info:
            case = self.band_gap_info.get('case', 'unknown')
            
            if case == 'semiconductor':
                v = self.band_gap_info['valence_top']
                c = self.band_gap_info['conduction_bottom']
                
                report_lines.append("分析结果: 半导体材料")
                report_lines.append("")
                
                report_lines.append("价带顶 (VBM):")
                report_lines.append(f"  能量: {v['energy']:.6f} eV")
                report_lines.append(f"  能带: {v['band_idx']+1}")
                report_lines.append(f"  k点索引: {v['k_idx']}")
                report_lines.append(f"  归一化路径坐标: {v['k_path']:.6f}")
                if v['symbol']:
                    report_lines.append(f"  高对称点: {v['symbol']}")
                report_lines.append(f"  分数坐标: [{v['k_coords'][0]:.4f}, {v['k_coords'][1]:.4f}, {v['k_coords'][2]:.4f}]")
                report_lines.append("")
                
                report_lines.append("导带底 (CBM):")
                report_lines.append(f"  能量: {c['energy']:.6f} eV")
                report_lines.append(f"  能带: {c['band_idx']+1}")
                report_lines.append(f"  k点索引: {c['k_idx']}")
                report_lines.append(f"  归一化路径坐标: {c['k_path']:.6f}")
                if c['symbol']:
                    report_lines.append(f"  高对称点: {c['symbol']}")
                report_lines.append(f"  分数坐标: [{c['k_coords'][0]:.4f}, {c['k_coords'][1]:.4f}, {c['k_coords'][2]:.4f}]")
                report_lines.append("")
                
                report_lines.append(f"带隙: {self.band_gap_info['band_gap']:.6f} eV")
                report_lines.append(f"带隙类型: {'直接' if self.band_gap_info['is_direct'] else '间接'}带隙")
                report_lines.append(f"材料类型: {self.band_gap_info['material_type']}")
            
            elif case in ['valence_top_at_fermi', 'conduction_bottom_at_fermi']:
                v = self.band_gap_info['valence_top']
                c = self.band_gap_info['conduction_bottom']
                
                if case == 'valence_top_at_fermi':
                    report_lines.append("分析结果: 价带顶刚好在费米能级上")
                else:
                    report_lines.append("分析结果: 导带底刚好在费米能级上")
                
                report_lines.append(f"说明: {self.band_gap_info.get('note', '')}")
                report_lines.append("")
                
                report_lines.append("价带顶 (VBM):")
                report_lines.append(f"  能量: {v['energy']:.6f} eV")
                report_lines.append(f"  能带: {v['band_idx']+1}")
                if v['symbol']:
                    report_lines.append(f"  位置: {v['symbol']}点")
                report_lines.append(f"  归一化路径坐标: {v['k_path']:.6f}")
                report_lines.append("")
                
                report_lines.append("导带底 (CBM):")
                report_lines.append(f"  能量: {c['energy']:.6f} eV")
                report_lines.append(f"  能带: {c['band_idx']+1}")
                if c['symbol']:
                    report_lines.append(f"  位置: {c['symbol']}点")
                report_lines.append(f"  归一化路径坐标: {c['k_path']:.6f}")
                report_lines.append("")
                
                report_lines.append(f"带隙: {self.band_gap_info['band_gap']:.6f} eV")
                report_lines.append(f"材料类型: {self.band_gap_info['material_type']}")
            
            elif case == 'metal':
                report_lines.append("分析结果: 金属性材料")
                report_lines.append(f"说明: {self.band_gap_info.get('note', '')}")
                if 'crossing_points' in self.band_gap_info:
                    report_lines.append("穿越费米能级的位置:")
                    for cp in self.band_gap_info['crossing_points']:
                        if cp['symbol']:
                            report_lines.append(f"  {cp['symbol']}点 (归一化路径坐标: {cp['k_path']:.6f})")
            
            elif case == 'metal_multiple_bands':
                report_lines.append("分析结果: 金属性材料（多条能带穿越费米能级）")
                report_lines.append(f"说明: {self.band_gap_info.get('note', '')}")
            
            elif case == 'possible_indirect_gap':
                report_lines.append("分析结果: 可能的间接带隙半导体")
                report_lines.append(f"说明: {self.band_gap_info.get('note', '')}")
                report_lines.append(f"带隙: {self.band_gap_info['band_gap']:.6f} eV")
                report_lines.append(f"材料类型: {self.band_gap_info['material_type']}")
            
            else:
                report_lines.append(f"分析结果: {self.band_gap_info.get('note', '未知情况')}")
        else:
            report_lines.append("未进行带隙分析")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 输出到控制台
        for line in report_lines:
            print(line)
        
        # 保存到文件
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                print(f"\n分析报告已保存到: {output_file}")
            except Exception as e:
                print(f"错误: 无法保存报告文件 - {e}")
        
        return report_lines

def main():
    """主函数"""
    try:
        # 设置命令行参数解析
        parser = argparse.ArgumentParser(
            description='ABINIT能带数据分析工具 - 增强版',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  python %(prog)s band_structure.agr
  python %(prog)s band_structure.agr --abi-file input.abi
  python %(prog)s band_structure.agr --abo-file output.abo
  python %(prog)s band_structure.agr --no-plot
  python %(prog)s band_structure.agr --interactive
  python %(prog)s band_structure.agr --output-prefix my_analysis
  python %(prog)s tbase3_5o_DS2_EBANDS.agr  # 自动寻找tbase3_5.abi和tbase3_5.abo文件
        """
        )
        
        parser.add_argument('agr_file', help='AGR格式的能带文件')
        parser.add_argument('--abi-file', help='ABINIT输入文件（用于获取晶格和高对称点信息）')
        parser.add_argument('--abo-file', help='ABINIT输出文件（用于获取费米能级）')
        parser.add_argument('--no-plot', action='store_true', help='不显示能带图')
        parser.add_argument('--interactive', action='store_true', help='使用交互模式进行带隙分析')
        parser.add_argument('--output-prefix', help='输出文件前缀')
        parser.add_argument('--dpi', type=int, default=300, help='图像分辨率（默认: 300）')
        
        args = parser.parse_args()
        
        # 检查AGR文件是否存在
        if not os.path.exists(args.agr_file):
            print(f"错误: AGR文件不存在 - {args.agr_file}")
            sys.exit(1)
        
        # 确定输出文件前缀
        if args.output_prefix:
            base_name = args.output_prefix
        else:
            base_name = os.path.splitext(os.path.basename(args.agr_file))[0]
        
        # 自动寻找相同前缀的.abi和.abo文件
        agr_basename = os.path.basename(args.agr_file)
        # 尝试提取任务前缀（比如tbase3_5o_DS2_EBANDS.agr -> tbase3_5）
        task_prefix = None
        
        # 方法1: 尝试不同的分隔符组合
        if '_' in agr_basename:
            # 例如: tbase3_5o_DS2_EBANDS.agr -> 检查tbase3_5o, tbase3_5, tbase3
            parts = agr_basename.split('_')
            for i in range(len(parts), 0, -1):
                # 检查不同长度的前缀
                for j in range(i, 0, -1):
                    potential_prefix = '_'.join(parts[:j])
                    # 移除可能的数字后缀（如tbase3_5o -> tbase3_5）
                    import re
                    cleaned_prefix = re.sub(r'[a-zA-Z]$', '', potential_prefix)
                    
                    potential_abi = f"{cleaned_prefix}.abi"
                    if os.path.exists(potential_abi):
                        task_prefix = cleaned_prefix
                        break
                if task_prefix:
                    break
        
        # 方法2: 如果方法1失败，尝试简单的前缀匹配
        if not task_prefix:
            # 例如: tbase3_5o_DS2_EBANDS.agr -> tbase3_5
            potential_prefix = agr_basename.split('_')[0]
            # 检查是否包含数字
            if any(c.isdigit() for c in potential_prefix):
                # 保留数字部分，移除字母后缀
                import re
                match = re.search(r'([a-zA-Z]+\d+)', potential_prefix)
                if match:
                    potential_prefix = match.group(1)
                    potential_abi = f"{potential_prefix}.abi"
                    if os.path.exists(potential_abi):
                        task_prefix = potential_prefix
        
        # 如果找到了任务前缀，自动设置.abi和.abo文件
        if task_prefix and not args.abi_file:
            potential_abi = f"{task_prefix}.abi"
            if os.path.exists(potential_abi):
                args.abi_file = potential_abi
                print(f"自动找到ABI文件: {args.abi_file}")
        
        if task_prefix and not args.abo_file:
            potential_abo = f"{task_prefix}.abo"
            if os.path.exists(potential_abo):
                args.abo_file = potential_abo
                print(f"自动找到ABO文件: {args.abo_file}")
        
        # 检查ABI文件是否存在（如果提供）
        if args.abi_file and not os.path.exists(args.abi_file):
            print(f"警告: ABINIT输入文件不存在 - {args.abi_file}")
            args.abi_file = None
        
        # 检查ABO文件是否存在（如果提供）
        if args.abo_file and not os.path.exists(args.abo_file):
            print(f"警告: ABINIT输出文件不存在 - {args.abo_file}")
            args.abo_file = None
        
        # 创建输出目录（如果不存在）
        output_dir = "abinit_band_output"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"输出目录: {output_dir}")
        
        # 创建分析器实例
        analyzer = AbinitBandAnalyzer(args.agr_file, args.abi_file)
        
        # 解析AGR文件
        if not analyzer.parse_agr_file():
            print("错误: AGR文件解析失败")
            sys.exit(1)
        
        # 解析ABI文件（如果提供）
        if args.abi_file:
            analyzer.parse_abi_file()
        
        # 不再从ABO文件中读取费米能级，只使用AGR文件中的数据
        # 解析ABO文件（仅用于额外信息，如Γ点能量）
        if args.abo_file:
            analyzer.parse_abo_file(args.abo_file)
        
        # 将能带数据相对于费米能级进行偏移
        print(f"\n正在将能带数据相对于费米能级进行偏移...")
        print(f"原始费米能级: {analyzer.efermi_original:.6f} eV")
        print(f"AGR能量零点是否已设置为费米能级: {analyzer.agr_energy_zero_is_efermi}")
        
        # 应用费米能级偏移逻辑
        # 1. 如果AGR文件中的能量零点已经是费米能级，无需再次偏移
        # 2. 否则，如果从ABO文件中读取了有效费米能级，使用它进行偏移
        if analyzer.agr_energy_zero_is_efermi:
            print(f"AGR文件中的能量零点已经是费米能级，无需再次偏移")
            analyzer.fermi_energy = 0.0
            print(f"当前费米能级: {analyzer.fermi_energy} eV")
        elif analyzer.efermi_original != 0.0:
            # 将所有能带数据减去费米能级，使费米能级变为0 eV
            for band_idx in range(len(analyzer.band_data)):
                analyzer.band_data[band_idx] = [energy - analyzer.efermi_original for energy in analyzer.band_data[band_idx]]
            print(f"已将能带数据相对于费米能级 {analyzer.efermi_original:.6f} eV 进行偏移")
            analyzer.fermi_energy = 0.0
            print(f"当前费米能级: {analyzer.fermi_energy} eV")
        else:
            analyzer.fermi_energy = 0.0
            print(f"未找到有效的费米能级，使用默认值: {analyzer.fermi_energy} eV")
        
        # 计算倒易空间
        analyzer.calculate_reciprocal_lattice()
        
        # 整理数据并保存为CSV
        csv_file = os.path.join(output_dir, f"{base_name}_bands.csv")
        analyzer.organize_band_data(output_csv=csv_file)
        
        # 在绘图前完成带隙分析
        print("\n" + "="*70)
        print("开始带隙分析（在绘图之前完成）")
        print("="*70)
        
        analyzer.analyze_band_gap(interactive_mode=args.interactive)
        
        # 绘制初始能带图（带隙分析结果会标注在图上）
        png_file = os.path.join(output_dir, f"{base_name}_bands.png")
        analyzer.plot_band_structure(
            output_image=png_file, 
            dpi=args.dpi, 
            show_plot=not args.no_plot
        )
        
        # 询问用户是否要调整费米能级
        if not args.no_plot:  # 只有在显示图像的情况下才提供交互调整
            adjust_choice = input("\n是否需要调整费米能级？(y/n): ").strip().lower()
            if adjust_choice == 'y':
                # 保存原始能带数据，用于调整费米能级时恢复
                original_band_data = [band.copy() for band in analyzer.band_data]
                
                # 调整费米能级
                final_fermi = analyzer.adjust_fermi_level()
                
                # 重新绘制调整后的能带图
                adjusted_png_file = os.path.join(output_dir, f"{base_name}_bands_adjusted.png")
                analyzer.plot_band_structure(
                    output_image=adjusted_png_file, 
                    dpi=args.dpi, 
                    show_plot=not args.no_plot
                )
                
                print(f"\n调整后的能带图已保存到: {adjusted_png_file}")
        
        # 生成分析报告
        report_file = os.path.join(output_dir, f"{base_name}_report.txt")
        analyzer.generate_summary_report(output_file=report_file)
        
        print("\n" + "="*70)
        print("分析完成！")
        print(f"输出目录: {output_dir}")
        print(f"数据文件: {csv_file}")
        print(f"能带图: {png_file}")
        print(f"分析报告: {report_file}")
        print("="*70)
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()