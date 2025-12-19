import numpy as np
import matplotlib.pyplot as plt

# 读取OSZICAR文件
file_path = 'OSZICAR'
energies = []
delta_energies = []

with open(file_path, 'r') as f:
    lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('DAV:'):
            # 提取DAV行的数据
            parts = line.split()
            # 第3列是E，第4列是dE
            e = float(parts[2])
            de = float(parts[3])
            energies.append(e)
            delta_energies.append(de)

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# 绘制总能量E
ax1.plot(range(len(energies)), energies, 'o-', markersize=4, linewidth=1, color='blue')
ax1.set_ylabel('Total Energy (E)', fontsize=12)
ax1.set_title('Energy Changes During Structure Optimization', fontsize=14)
ax1.grid(True, alpha=0.3)
# 确保纵坐标以指数形式显示
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# 确保ytick使用科学计数法
ax1.yaxis.get_major_formatter().set_scientific(True)
ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))

# 绘制能量变化dE
ax2.plot(range(len(delta_energies)), delta_energies, 'o-', markersize=4, linewidth=1, color='red')
ax2.set_xlabel('Iteration Step', fontsize=12)
ax2.set_ylabel('Energy Change (dE)', fontsize=12)
ax2.grid(True, alpha=0.3)
# 确保纵坐标以指数形式显示
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# 确保ytick使用科学计数法
ax2.yaxis.get_major_formatter().set_scientific(True)
ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('energy_optimization.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()