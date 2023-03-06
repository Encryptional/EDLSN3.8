import os
dirname = ['/home/zhangbin/DNA/Datasets/Datasets/RNA-495_Train', '/home/zhangbin/DNA/Datasets/Datasets/RNA-117_Test', '/home/zhangbin/DNA/Datasets/Datasets/DNA-573_Train', '/home/zhangbin/DNA/Datasets/Datasets/DNA-129_Test']#dirname用于存放path目录下所有目录的路径，即文件夹路径
for i in range(len(dirname)):
    count = 0
    seqname = os.listdir(dirname[i])
    savepath = 'cd '+dirname[i]+';ASAquick'#'cd '+dirname[i]+'表示要将结果保存到的路径位置，ASAquick为执行命令
    for j in range(len(seqname)):
                   seq_file = dirname[i]+'/'+seqname[j]
                   os.system('%s %s'%(savepath, seq_file))
                   count = count + 1
    print(count)

