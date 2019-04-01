import time

from Edg_extraction import Edge_extraction

start = time.clock()
imgfile = 'E:\\file\\YS@J_0@RZ6~242D1S`1ICCD.JPG'
img = Edge_extraction(imgfile, 'Laplace')
end = time.clock()
print(end - start)
img.show()
