#!/usr/bin/env
# coding: utf-8

import os

def run():
    #将gro文件第二列的原子编号删除
    path = os.listdir(os.getcwd())
    
    for i in path:
        if '.gro' in i:
            with open(i) as f:
                g = f.readlines()

            #根据*.gro文件中的原子数，生成一个posre的位置限制*.itp文件
            atom = []
            for k in g[2:-1]:
                word = k.split()
                if word[2][0] != 'H':
                    atom.append(word[3])
                    
            new = []
            new.append('[ position_restraints ]'+'\n')
            new.append('; atom    type    fx    fy    fz'+'\n')
            for j in atom:
                if int(j) < 10:
                    new.append('    '+' '+j+'   1    1000    1000    1000'+'\n')
                else:
                    new.append('    '+j+'   1    1000    1000    1000'+'\n')

            temp = os.path.splitext(i)[0]
            file = temp.split('_')[0]
            fileName = file + '.itp'
            
            with open(fileName,'w') as p:
                p.writelines(new)
        
        
if __name__ == '__main__':
    run()

            

        
            
            
