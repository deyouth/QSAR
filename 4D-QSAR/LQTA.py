#!/usr/bin/env
# coding: utf-8

import os,re,math

def Distance(r1, r2):
    d = math.sqrt(math.pow((r1[0] - r2[0]), 2)
            + math.pow((r1[1] - r2[1]) ,2) + math.pow((r1[2] - r2[2]), 2))
    return d

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def triplewise(iterable):
    a = iter(iterable)
    return zip(a, a, a)

class MatrixGenerate():

    def __init__(self, fileGro, fileTop, fileItp):
        self.setX(fileGro)
        self.atomsTypes(fileTop)
        self.loadConstants(fileItp)
        self.loadAP()
        self.determineConstants()

    def setX(self, fileName):
        with open(fileName) as f:
            input = f.readlines()
        currentLine = 1
        line = input[currentLine]

        x = []
        y = []
        z = []
        self.numberElements = int(input[currentLine])
        self.c6 = []
        self.c12 = []

        while True:
            for i in range(self.numberElements):
                currentToken = 3
                currentLine += 1
                line = input[currentLine]

                #tokens = re.findall(r"[\w\.']+", line)
                tokens = line.split()
                x.insert(i, tokens[currentToken])
                currentToken += 1
                y.insert(i, tokens[currentToken])
                currentToken += 1
                z.insert(i, tokens[currentToken])

            currentLine += 1
            if len(input) > currentLine + 1:
                currentLine += 1
            else:
                break

            currentLine += 1

        self.m = len(x)
        self.X = [[0 for self.X in range(3)] for self.X in range(self.m)]

        self.minimos = [float(x[0]) * 10,float(y[0]) * 10,float(z[0]) * 10]
        self.maximos = [float(x[0]) * 10,float(y[0]) * 10,float(z[0]) * 10]
        for i in range(self.m):
            self.X[i][0] = float(x[i]) * 10
            self.X[i][1] = float(y[i]) * 10
            self.X[i][2] = float(z[i]) * 10
            self.minimos[0] = min(self.minimos[0],self.X[i][0])
            self.minimos[1] = min(self.minimos[1],self.X[i][1])
            self.minimos[2] = min(self.minimos[2],self.X[i][2])
            self.maximos[0] = max(self.maximos[0],self.X[i][0])
            self.maximos[1] = max(self.maximos[1],self.X[i][1])
            self.maximos[2] = max(self.maximos[2],self.X[i][2])

    def atomsTypes(self, fileName):
        with open(fileName) as f:
            input = f.readlines()

        self.types = []
        self.cargos = []

        currentLine = 0
        line = input[currentLine]

        tokens = re.findall(r"[\w\.']+", line)
        token = ""
        
        while token != "atoms":
            currentToken = 0
            currentLine += 1
            line = input[currentLine]

            if line != "":
                tokens = re.findall(r"[\w\.\-\+\[\]\;\n\=']+", line)

                if tokens[currentToken] == "[":
                    currentToken += 1
                    token = tokens[currentToken]

        currentLine += 2
        line = input[currentLine]
        for i in range(self.numberElements):
            tokens = re.findall(r"[\w\.\-\+\[\]\;\n\=']+", line)
            currentToken = 1
            self.types.insert(i, tokens[currentToken])
            currentToken += 5
            self.cargos.insert(i, tokens[currentToken])
            currentLine += 1
            line = input[currentLine]

    def loadConstants(self,fileName):
        with open(fileName) as f:
            input = f.readlines()

        ttype = []
        sigma = []
        epsilon = []
        currentLine = 0
        line = input[currentLine]
        index = 1
        
        while line != "[ atomtypes ]\n": 
            currentLine += 1
            line = input[currentLine]        
        currentLine += 1
        
        while len(input) > currentLine +1:
            currentToken = 0
            currentLine += 1
            line = input[currentLine]
            tokens = re.findall(r"[\w\.\-\+']+", line)
            ttype.insert(index, tokens[currentToken])
            currentToken += 4
            sigma.insert(index, tokens[currentToken])
            currentToken += 1
            epsilon.insert(index, tokens[currentToken])
            index += 1

        nttypes = len(ttype)
        self.typeConstants = []
        self.constantc6 = []
        self.constantc12 = []

        for i in range(nttypes):
            self.typeConstants.insert(i, ttype[i])

            self.constantc6.insert(i, 4.0 * float(epsilon[i])
                                    * (float(sigma[i]) ** 6))
            self.constantc12.insert(i, 4.0 * float(epsilon[i])
                                    * (float(sigma[i]) ** 12))

    def loadAP(self):
        with open("AtomProva.atp") as f:
            input = f.readlines()

        self.ap = []
        self.cargosap = []
        self.c6ap = []
        self.c12ap = []

        currentLine = 1
        line = input[currentLine]
        index = 0

        while len(input) > currentLine + 1:
            currentToken = 0
            currentLine += 1
            line = input[currentLine]
            tokens = re.findall(r"[\w\.\-\+\(\)\=']+", line)
            self.ap.insert(index, tokens[currentToken])
            currentToken += 1
            self.cargosap.insert(index, float(tokens[currentToken]))
            currentToken += 1
            self.c6ap.insert(index, float(tokens[currentToken]))
            currentToken += 1
            self.c12ap.insert(index, float(tokens[currentToken]))
            index += 1

    def search(self, vector, element):
        nElem = len(vector)

        for i in range(nElem):
            if element == vector[i]:
                return i

        return -1

    def determineConstants(self):
        for i in range(self.numberElements):
            index = self.search(self.typeConstants, self.types[i])
            self.c6.insert(i, self.constantc6[index])
            self.c12.insert(i, self.constantc12[index])
        
    def gridGenerate(self, dimX, dimY, dimZ, atp, x0, y0, z0, step):
        self.DimX = dimX
        self.DimY = dimY
        self.DimZ = dimZ
        self.natp = len(atp)

        f = 138.935485
        nframes = self.m / self.numberElements
        self.gridCoulomb = [[[[0 for x in range(self.natp)] for x in range(self.DimZ)]
                            for x in range(self.DimY)] for x in range(self.DimX)]

        self.gridLJ = [[[[0 for x in range(self.natp)] for x in range(self.DimZ)]
                        for x in range(self.DimY)] for x in range(self.DimX)]

        print(("Dimensions %d X %d X %d\n" % (self.DimX,self.DimY,self.DimZ)))
        for h in range(self.natp):
            elem = self.search(self.ap, atp[h])
            q1 = self.cargosap[elem]
            c6a = self.c6ap[elem]
            c12a = self.c12ap[elem]
            Vlj = 0
            Vc = 0
            npontos = 0
            #r1 = []
            r1 = [0.0,0.0,0.0]
	            
            print(("Calculating %s probe\n" % atp[h]))
            for i in range(self.DimX):
                r1[0] = i*step+x0
                for j in range(self.DimY):
                    r1[1] = j*step+y0
                    for k in range(self.DimZ):
                        r1[2] = k*step+z0
                        Vlj = 0
                        Vc = 0
                        npontos += 1
                        for l in range(self.m):
                            r = Distance(r1, self.X[l]) / 10
                            index = l % self.numberElements
                            c6ij = math.sqrt(c6a * self.c6[index])
                            c12ij = math.sqrt(c12a * self.c12[index])

                            if r != 0:
                                Vlj = Vlj + (c12ij / (math.pow(r, 12))) - (c6ij / (math.pow(r, 6)))
                                Vc = Vc + f * float(q1) * float(self.cargos[index]) / r
                            else:
                                Vlj = float("inf")
                                Vc = float("inf")

                        self.gridCoulomb[i][j][k][h] = Vc / nframes
                        self.gridLJ[i][j][k][h] = Vlj / math.sqrt(nframes)

    def getMatrix(self, optionGrid):
        result = ""
        for i in range(self.DimX):
            for j in range(self.DimY):
                for k in range(self.DimZ):
                    for l in range(self.natp):
                        if optionGrid == "C":
                            result += "%g\t" % (self.gridCoulomb[i][j][k][l])
                        elif optionGrid == 'L':
                            result += "%g\t" % (self.gridLJ[i][j][k][l])
                        else:
                            pass
        return result

class GridGenerate():

    def __init__(self, coordinates, dimensions, atp, files, step):
        
        dataFile = open(files).read().splitlines()
        matrices = [] 

        minimos = [999999.0,999999.0,999999.0]
        maximos = [-999999.0,-999999.0,-999999.0]
        #for fileGro, fileItp, fileTop in pairwise(dataFile):
        for fileGro, fileTop, fileItp in triplewise(dataFile):
            matrix = MatrixGenerate(fileGro, fileTop, fileItp)
            minimos[0] = min(minimos[0],matrix.minimos[0])
            minimos[1] = min(minimos[1],matrix.minimos[1])
            minimos[2] = min(minimos[2],matrix.minimos[2])
            maximos[0] = max(maximos[0],matrix.maximos[0])
            maximos[1] = max(maximos[1],matrix.maximos[1])
            maximos[2] = max(maximos[2],matrix.maximos[2])
            matrices.append(matrix)

        if coordinates != ():
            x0, y0, z0 = coordinates
        else:
            x0 = int(minimos[0])-5
            y0 = int(minimos[1])-5
            z0 = int(minimos[2])-5
            
        if dimensions != (): 
            dim_x, dim_y, dim_z = dimensions
        else:
            dim_x = int(maximos[0]-minimos[0])+10
            dim_y = int(maximos[1]-minimos[1])+10
            dim_z = int(maximos[2]-minimos[2])+10
            
        if not step == 1: 
            I = int((dim_x/step)+(1/step-1))
            J = int((dim_y/step)+(1/step-1))
            K = int((dim_z/step)+(1/step-1))
        else:
            I = dim_x + 1
            J = dim_y + 1
            K = dim_z + 1

        n = len(atp)
        coulomb = ""
        lj = ""

        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(n):
                        value_x = i*step+x0
                        value_y = j*step+y0
                        value_z = k*step+z0
                        coulomb += "%.2f_%.2f_%.2f_%s_C: \t" % (value_x, value_y,
                                                                value_z, atp[l])

                        lj += "%.2f_%.2f_%.2f_%s_LJ: \t" % (value_x, value_y,
                                                            value_z, atp[l])
        self.output = coulomb + lj
        
        for matrix in matrices:
            matrix.gridGenerate(I, J, K, atp, x0, y0, z0, step)
            valuesCoulomb = matrix.getMatrix("C")
            valuesLj = matrix.getMatrix("L")
            self.output += "\n" + valuesCoulomb + valuesLj            
            
    def saveGrid(self,output):
        arq = open(output, "w")
        arq.write(self.output)
        arq.close()
        
def itp_killer():
    '''更改itp文件格式'''
    path = os.listdir(os.getcwd())

    for i in path:
        if '.itp' in i:
            with open(i) as f:
                inp = f.readlines()

            new = []
            currentLine = 0
            line = inp[currentLine]

            new.append(inp[currentLine])
            

            while len(inp) > currentLine +1:
                currentLine += 1
                line = inp[currentLine]
                tokens = re.findall(r"[\w\.\-\+']+", line)
        
                if len(tokens) > 6:
                    del tokens[1]
                a = '   '.join(tokens) + '\n'  
                new.append(a)
                
            q = open(i,'w')
            q.writelines(new)
            q.close()

def calcBox():
    '''BOX计算器'''

    with open('box.txt') as f:
        line = f.readlines()

    num = []

    for i in line[1:]:
        num.append(i.split()[1:])

        
        

    minX,minY,minZ,maxX,maxY,maxZ = float(num[0][0]),float(num[0][1]),\
                                    float(num[0][2]),float(num[1][0]),\
                                    float(num[1][1]),float(num[1][2])

    def minCal(num):
        return round((num - 0.15) * 10)

    def maxCal(num):
        return round((num + 0.15) * 10)

    X = maxCal(maxX) - minCal(minX)
    Y = maxCal(maxY) - minCal(minY)
    Z = maxCal(maxZ) - minCal(minZ)

    small_X = minCal(minX)
    small_Y = minCal(minY)
    small_Z = minCal(minZ)

    print('lqtagridpy --mols list.txt -c %s %s %s -d %s %s %s \
    -a NH3+ -s 1 -o matrix.txt\n' %(small_X,small_Y,small_Z,X,Y,Z))

    return small_X,small_Y,small_Z,X,Y,Z

def lqta(mols, coordinates, dimensions, atom, step,output):

    grid = GridGenerate(
        coordinates,
        dimensions,
        atom,
        mols,
        step
    )
    grid.saveGrid(output)


def runLQTA():
    #更改itp格式
    itp_killer()

    #自动计算BOX参数
    Box = calcBox()
    coordinates = Box[:3]
    dimensions = Box[3:]

    #探针类型
    atom = ['NH3+']

    #文件列表
    mols = 'list.txt'

    step = 1
    
    #输出文件名称
    output = 'matrix.txt'
    
    #运行LQTAgrid
    lqta(mols, coordinates, dimensions, atom, step,output)


if __name__ == '__main__':
    runLQTA()
    
