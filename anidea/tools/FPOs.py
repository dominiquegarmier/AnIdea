import numpy as np

# Calculate FPOs of NN

tanhflops = 15
reluflops = 2
sineflops = 15

def matmulf(n,m,k, isparallel):
    if isparallel:
        return 2*m-1
    else:
        return (2*m-1)*n*k

def vecaddf(n, isparallel):
    if isparallel:
        return 1
    else:
        return n

def actfunc(n, afunc, isparallel):
    if isparallel:
        return afunc
    else:
        return n*afunc

def layerf(n, m, afunc, isparallel):
	# weight matrix: m times n 
	# (m, n)*(n, 1)

	flops = 0

	flops += matmulf(n, m, 1, isparallel)
	flops += vecaddf(m, isparallel)
	flops += actfunc(m, afunc, isparallel)

	return flops

def ffdnnf(nnarc, afunc, isparallel):

	flops = 0
	prevlayer = nnarc[0]
	for width in nnarc[1:]:

		flops += layerf(prevlayer, width, afunc, isparallel)
		prevlayer = width

	return flops

def p2f(n, isparallel):
    if isparallel:
        return 5
    else:
        return 13*n

def paramf(n, nnarc, afunc, isparallel):

    flops = 0
    prevlayer = 2
    for width in nnarc:

        flops += layerf(prevlayer, n*width, afunc, isparallel)
        prevlayer = n*width

    flops += layerf(prevlayer, n, afunc, isparallel)

    return flops

def nnf(NNarc, nnafunc, paramafunc, isparallel):

    flops = 0
    ffdlayers = []
    for layer in NNarc:
        ffdlayers.append(layer[0])

    flops += ffdnnf(ffdlayers, nnafunc, isparallel)

    paramflops = 0
    prevwidth = 0
    for layer in NNarc[1:]:

        matrixn = prevwidth*layer[0]
        biasn = layer[0]

        type_ = layer[1]
        if type_ == 'p2':

            flps = p2f(matrixn + biasn, isparallel)

            if isparallel and paramflops < flps:
                paramflops = flps
            elif not isparallel:
                paramflops += flps

        else:

            flps = paramf(matrixn, type_, paramafunc, isparallel)
			
            if isparallel and paramflops < flps:
                paramflops = flps
            elif not isparallel:
                paramflops += flps

            flps = 0

            flps = paramf(biasn, type_, paramafunc, isparallel)
			
            if isparallel and paramflops < flps:
                paramflops = flps
            elif not isparallel:
                paramflops += flps
            flps = 0

    flops += paramflops
    return flops



nnarc = [[1, 'input layer'],
    [5, [2, 4, 4, 2]],
    [10, [2, 4, 4, 2]],
    [15, [2, 4, 4, 2]],
    [20, [2, 4, 4, 2]],
    [20, [2, 4, 4, 2]],
    [20, [2, 4, 4, 2]],
    [20, [2, 4, 4, 2]],
    [20, [2, 4, 4, 2]],
    [15, [2, 4, 4, 2]],
    [10, [2, 4, 4, 2]],
    [5, [2, 4, 4, 2]],
    [1, 'p2']]

print(nnf(nnarc, tanhflops, reluflops, True), nnf(nnarc, tanhflops, reluflops, False))
