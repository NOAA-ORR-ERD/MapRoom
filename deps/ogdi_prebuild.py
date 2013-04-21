import os
import shutil
libiconv_dir = '../../libiconv/workspace'
proj_dir = '../../PROJ.4/workspace'

#if not os.path.exists(libiconv_dir):
#    shutil.copytree('../libiconv-1.11', libiconv_dir)

if not os.path.exists(proj_dir):
    shutil.copytree('../proj-4.7.0', proj_dir)