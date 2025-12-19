# 读取OSZICAR文件，检查数据分布
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
            e = float(parts[2])
            de = float(parts[3])
            energies.append(e)
            delta_energies.append(de)

print("能量数据统计：")
print(f"E的数量：{len(energies)}")
print(f"E的最小值：{min(energies):.10e}")
print(f"E的最大值：{max(energies):.10e}")
print(f"E的范围：{max(energies) - min(energies):.10e}")

print("\ndE数据统计：")
print(f"dE的数量：{len(delta_energies)}")
print(f"dE的最小值：{min(delta_energies):.10e}")
print(f"dE的最大值：{max(delta_energies):.10e}")
print(f"dE的范围：{max(delta_energies) - min(delta_energies):.10e}")

# 查看前几个和后几个dE值
print("\ndE前5个值：")
for de in delta_energies[:5]:
    print(f"{de:.10e}")

print("\ndE后5个值：")
for de in delta_energies[-5:]:
    print(f"{de:.10e}")