#!/usr/bin/env python3
"""
一键绘制VASP结构优化能量变化过程脚本
整合：字体自动设置 + log指数坐标 + 跳点功能 + 智能解析
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.ticker import FuncFormatter, ScalarFormatter

def setup_chinese_font():
    """自动设置中文字体支持"""
    try:
        # 直接使用英文字体，避免中文显示问题
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("✓ 已设置英文字体，避免中文显示问题")
        return True
    except Exception as e:
        print(f"字体设置警告: {e}")
        return False

def robust_parse_oszicar(file_path):
    """稳健解析OSZICAR文件"""
    ion_steps = []
    current_steps = []
    
    print(f"解析文件: {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 1. 电子步行 (包含算法标签)
            if (('DAV:' in line) or ('RMM:' in line) or ('EDIAG:' in line) or 
                ('CG:' in line) or (line.startswith('DAV:') or line.startswith('RMM:'))):
                
                # 提取所有科学计数法数字
                sci_nums = re.findall(r'[+-]?\d+\.\d+[Ee][+-]\d+', line)
                if len(sci_nums) >= 2:
                    try:
                        step_data = {
                            'E': float(sci_nums[1]),  # 第二个通常是能量E
                            'dE': float(sci_nums[2]) if len(sci_nums) >= 3 else 0.0
                        }
                        current_steps.append(step_data)
                    except:
                        pass
            
            # 2. 离子步结束行 (F= 或 E0=)
            elif 'F=' in line or 'E0=' in line:
                if current_steps:
                    ion_steps.append(current_steps.copy())
                    current_steps = []
    
    # 处理最后一步
    if current_steps:
        ion_steps.append(current_steps)
    
    # 备用解析：如果上面没解析到，尝试通用解析
    if not ion_steps:
        print("尝试备用解析...")
        with open(file_path, 'r') as f:
            content = f.read()
            # 找到所有形如 -0.123456E+02 的数字
            all_numbers = re.findall(r'[+-]?\d+\.\d+[Ee][+-]\d+', content)
            nums = [float(n) for n in all_numbers]
            
            # 简单分组：每4个数字为一组
            for i in range(0, len(nums), 4):
                if i+1 < len(nums):
                    step_data = {'E': nums[i+1]}
                    if i+2 < len(nums):
                        step_data['dE'] = nums[i+2]
                    current_steps.append(step_data)
            
            if current_steps:
                # 简单分组为离子步：每10个电子步为一个离子步
                step_size = 10
                for i in range(0, len(current_steps), step_size):
                    ion_steps.append(current_steps[i:i+step_size])
    
    return ion_steps

def create_log_formatter(original_values):
    """创建log坐标的自定义格式化器，显示原始带符号的值"""
    def format_func(value, tick_number):
        # 在原始值中寻找最接近的值
        if not hasattr(format_func, 'abs_values'):
            format_func.abs_values = [abs(v) for v in original_values]
            format_func.orig_values = original_values
        
        # 避免除零
        if value <= 0:
            return "0"
        
        # 找到最接近的绝对值对应的原始值
        idx = min(range(len(format_func.abs_values)), 
                 key=lambda i: abs(format_func.abs_values[i] - value))
        
        # 格式化显示
        orig_val = format_func.orig_values[idx]
        if abs(orig_val) >= 100:
            return f"{orig_val:.0f}"
        elif abs(orig_val) >= 10:
            return f"{orig_val:.1f}"
        elif abs(orig_val) >= 1:
            return f"{orig_val:.2f}"
        elif abs(orig_val) >= 0.1:
            return f"{orig_val:.3f}"
        else:
            return f"{orig_val:.2e}"
    
    return format_func

def plot_with_options(ion_steps, output_base="energy_convergence"):
    """绘制能量收敛图（整合所有功能）"""
    if not ion_steps:
        print("错误: 无数据可绘制！")
        return False
    
    print(f"\n{'='*60}")
    print("找到数据:")
    print(f"  离子步数: {len(ion_steps)}")
    for i, steps in enumerate(ion_steps):
        print(f"  离子步{i+1}: {len(steps)}个电子步")
    
    # 用户交互
    print(f"\n{'='*60}")
    
    # 1. 选择纵坐标类型
    while True:
        choice = input("选择纵坐标类型:\n  1. 线性坐标 (默认)\n  2. 对数坐标 (推荐初始值很大时)\n请选择 [1/2]: ").strip()
        if choice in ['', '1', '线性', 'linear']:
            use_log_scale = False
            print("使用线性坐标")
            break
        elif choice in ['2', '对数', 'log']:
            use_log_scale = True
            print("使用对数坐标")
            break
        else:
            print("请输入 1 或 2")
    
    # 2. 选择跳点数
    max_skip = min(len(s) for s in ion_steps) - 1
    skip_points = 0
    if max_skip > 0:
        while True:
            skip_input = input(f"跳过每个离子步前几个点? (0-{max_skip}, 默认0): ").strip()
            if skip_input == '':
                skip_points = 0
                break
            try:
                skip_points = int(skip_input)
                if 0 <= skip_points <= max_skip:
                    break
                else:
                    print(f"请输入0到{max_skip}之间的数字")
            except:
                print("请输入有效数字")
    
    # 准备数据（应用跳点）
    all_E, all_dE, x_positions = [], [], []
    ion_boundaries = []
    x_counter = 0
    
    for ion_idx, steps in enumerate(ion_steps):
        # 跳过前N个点
        display_steps = steps[skip_points:] if skip_points < len(steps) else []
        if not display_steps:
            continue
        
        start_idx = x_counter
        
        for step in display_steps:
            if 'E' in step:
                all_E.append(step['E'])
                x_positions.append(x_counter)
                x_counter += 1
            
            if 'dE' in step and len(all_dE) < len(all_E):
                all_dE.append(abs(step['dE']))
        
        end_idx = x_counter - 1
        if end_idx >= start_idx:
            ion_boundaries.append({
                'start': start_idx,
                'end': end_idx,
                'ion_num': ion_idx + 1,
                'original_steps': len(steps),
                'display_steps': len(display_steps)
            })
    
    # 检查数据
    if len(all_E) < 2:
        print(f"错误: 跳点后只剩{len(all_E)}个点，无法绘图！")
        return False
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                            gridspec_kw={'height_ratios': [2, 1]})
    ax1, ax2 = axes
    
    # ========== 图1: Total Energy ==========
    line1 = ax1.plot(x_positions, all_E, 'b-', linewidth=2, 
                    marker='o', markersize=4, markevery=0.1,
                    label=f'Total Energy (E)', zorder=5)[0]
    
    # 设置坐标轴
    ax1.set_ylabel('Total Energy (eV)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax1.legend(loc='best', fontsize=11)
    
    # 对数坐标处理
    if use_log_scale:
        # 取绝对值，避免负值问题
        E_abs = [abs(e) for e in all_E]
        E_abs = [max(e, 1e-10) for e in E_abs]  # 避免0
        
        # 重绘为对数坐标
        ax1.clear()
        ax1.plot(x_positions, E_abs, 'b-', linewidth=2, 
                marker='o', markersize=4, markevery=0.1,
                label=f'|Total Energy| (|E|)', zorder=5)
        ax1.set_yscale('log')
        
        # 自定义格式化器，显示原始带符号的值
        formatter = FuncFormatter(create_log_formatter(all_E))
        ax1.yaxis.set_major_formatter(formatter)
        
        ax1.set_ylabel('|Total Energy| (eV, log scale)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', zorder=1, which='both')
        ax1.legend(loc='best', fontsize=11)
    
    # 标记离子步边界
    for boundary in ion_boundaries:
        start, end = boundary['start'], boundary['end']
        ion_num = boundary['ion_num']
        
        # 垂直线
        if end < len(x_positions) - 1:
            ax1.axvline(x=end + 0.5, color='red', linestyle=':', 
                       alpha=0.6, linewidth=1.5, zorder=2)
        
        # 标签
        mid_x = (start + end) / 2
        y_lim = ax1.get_ylim()
        
        if use_log_scale:
            log_min, log_max = np.log10(y_lim[0]), np.log10(y_lim[1])
            label_y = 10**(log_min + 0.05 * (log_max - log_min))
        else:
            label_y = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])
        
        ax1.text(mid_x, label_y, f'Ion Step {ion_num}', 
                ha='center', va='bottom', fontsize=10, zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='yellow', alpha=0.7))
    
    # ========== 图2: Energy Change (Always log scale) ==========
    if all_dE and len(all_dE) == len(all_E):
        # 确保长度一致
        dE_to_plot = all_dE[:len(all_E)]
        
        ax2.semilogy(x_positions, dE_to_plot, 'r-', linewidth=1.5,
                    marker='s', markersize=3, markevery=0.1,
                    label='|Energy Change| (|dE|)', zorder=5)
        
        # 收敛阈值线
        thresholds = [
            (1e-3, 'green', '1 meV'),
            (1e-4, 'orange', '0.1 meV'), 
            (1e-5, 'red', '0.01 meV'),
            (1e-6, 'purple', '0.001 meV')
        ]
        
        for thresh, color, label in thresholds:
            ax2.axhline(y=thresh, color=color, linestyle='--',
                       alpha=0.5, linewidth=1, label=label, zorder=1)
        
        ax2.set_xlabel('Cumulative Electronic Steps', fontsize=12, fontweight='bold')
        ax2.set_ylabel('|ΔE| (eV)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', which='both', zorder=1)
        
        # 智能图例
        handles, labels = ax2.get_legend_handles_labels()
        if len(handles) > 4:  # 只显示主要项
            ax2.legend(handles[:4], labels[:4], loc='upper right', 
                      fontsize=9, ncol=2)
        else:
            ax2.legend(loc='upper right', fontsize=9)
    
    # ========== Title and Layout ==========
    title_parts = ['VASP Structure Optimization Energy Convergence']
    if skip_points > 0:
        title_parts.append(f'(Skip {skip_points} points)')
    if use_log_scale:
        title_parts.append('- Log Scale')
    
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    img_file = f"{output_base}.png"
    plt.savefig(img_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    # 显示
    plt.show()
    
    # ========== 保存数据 ==========
    csv_file = f"{output_base}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("离子步,电子步,X位置,能量_E(eV),能量变化_dE(eV)\n")
        
        point_idx = 0
        for boundary in ion_boundaries:
            ion_num = boundary['ion_num']
            start, end = boundary['start'], boundary['end']
            
            for local_step in range(end - start + 1):
                if point_idx < len(all_E):
                    e_val = all_E[point_idx]
                    dE_val = all_dE[point_idx] if point_idx < len(all_dE) else 0
                    f.write(f"{ion_num},{local_step+1},{x_positions[point_idx]},{e_val},{dE_val}\n")
                    point_idx += 1
    
    # ========== 输出统计 ==========
    print(f"\n{'='*60}")
    print("绘图完成!")
    print(f"  图像文件: {img_file}")
    print(f"  数据文件: {csv_file}")
    print(f"\n统计信息:")
    print(f"  总离子步数: {len(ion_steps)}")
    print(f"  总电子步数: {sum(len(s) for s in ion_steps)}")
    print(f"  显示点数: {len(all_E)} (跳过前{skip_points}点)")
    
    if all_E:
        print(f"  能量范围: {min(all_E):.6f} 到 {max(all_E):.6f} eV")
        print(f"  初始能量: {all_E[0]:.6f} eV")
        print(f"  最终能量: {all_E[-1]:.6f} eV")
        print(f"  总能量变化: {all_E[0] - all_E[-1]:.6f} eV")
    
    print(f"{'='*60}")
    
    return True

def main():
    """主函数"""
    print("="*70)
    print("VASP结构优化能量分析工具 v2.0")
    print("功能: 自动解析OSZICAR + 智能绘图 + log坐标 + 跳点功能")
    print("="*70)
    
    # 设置字体
    setup_chinese_font()
    
    # 获取文件
    if len(sys.argv) > 1:
        oszicar_file = sys.argv[1]
    else:
        oszicar_file = "OSZICAR"
    
    # 检查文件
    if not os.path.exists(oszicar_file):
        print(f"错误: 文件不存在 '{oszicar_file}'")
        print("\n使用方法:")
        print("  1. 将脚本放在VASP计算目录")
        print("  2. 运行: python plot_vasp_energy.py")
        print("  3. 或指定文件: python plot_vasp_energy.py /path/to/OSZICAR")
        return
    
    try:
        # 解析
        print(f"\n正在处理: {oszicar_file}")
        ion_steps = robust_parse_oszicar(oszicar_file)
        
        if not ion_steps:
            print("无法解析文件，请检查格式")
            return
        
        # 绘图
        success = plot_with_options(ion_steps)
        
        if success:
            print("\n✓ 所有功能执行完成!")
        else:
            print("\n✗ 绘图过程中出现问题")
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()