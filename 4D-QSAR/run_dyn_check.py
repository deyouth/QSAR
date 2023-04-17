#!/usr/bin/env
# coding: utf-8

import os

count = 1

while count != 6:
    fail_dir = []
    reDyn = []
    path = os.listdir(os.getcwd())
    #检查run_dyn_m运行过后的每个文件夹中是否有最终输出文件pmd.gro
    for i in path:
        if os.path.isdir(i):
            if os.path.exists(os.getcwd()+'/'+i+'/'+'pmd.gro') == 0:
                #若发现有运行失败的，则将输入文件的文件名保存于reDyn.txt中
                reDyn.append(i+'.log'+'\n')
                reDyn.append(i+'_GMX.gro'+'\n')
                reDyn.append(i+'_GMX.top'+'\n')
                fail_dir.append(i+'\n')
                
    if reDyn == []:
        break
    
    with open('reDyn.txt','w') as f:
        f.writelines(reDyn)

    with open('fail_dir.txt','w') as f:
        f.writelines(fail_dir)

    #更改din_script_m中cg步骤的-nt参数，重新进行run_dyn_m
    nt = "sed -i 13s/.*/'gmx mdrun -s cg.tpr -o cg.trr -c pr.gro -g cg.log -nt "+str(count * 2)+"'/ ~/QSAR_KING/files/din_script_m"
    os.system(nt)
      
    os.system('re_Dyn')
    
    count += 1

defaultNT = "sed -i 13s/.*/'gmx mdrun -s cg.tpr -o cg.trr -c pr.gro -g cg.log -pin on'/ ~/QSAR_KING/files/din_script_m"

os.system(defaultNT)
