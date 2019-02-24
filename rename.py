import os

path = '/home/xddz/wharf_images/temp/failure'
imlist = os.listdir('/home/xddz/wharf_images/temp/failure')
for i in range(len(imlist)):
    os.rename(os.path.join(path, imlist[i]), os.path.join(path, '%05d.jpg'%i))