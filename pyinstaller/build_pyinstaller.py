#!/usr/bin/env python
import os
import sys
import shutil
from subprocess import Popen, PIPE

if sys.platform == 'win32':
    win = True
    mac = False
    exe = ".exe"
    onefile = True
elif sys.platform == 'darwin':
    win = False
    mac = True
    exe = ".app"
    onefile = True
else:  # linux
    win = False
    mac = False
    exe = ""
    onefile = False

if "-d" in sys.argv:
    onefile = False


exec(compile(open("../maproom/Version.py").read(), "../maproom/Version.py", 'exec'))

from subprocess import Popen, PIPE

def run(args):
    p = Popen(args, stdout=PIPE, bufsize=1)
    with p.stdout:
        for line in iter(p.stdout.readline, b''):
            print(line, end=' ')
    p.wait()

# can't use MapRoom because MapRoom is a directory name and the filesystem is
# case-insensitive
if onefile:
    build_target="MapRoom_build"
else:
    build_target="MapRoom_folder"
build_app = "dist/" + build_target + exe

# target app will be renamed
final_target="MapRoom"
dest_dir = "../dist-%s" % VERSION
final_app = final_target + exe
dest_app = "%s/%s" % (dest_dir, final_app)
final_exe = "%s-%s-win.exe" % (final_target, VERSION)
final_zip = "%s-%s-darwin.tbz" % (final_target, VERSION)
dest_exe = "%s/%s" % (dest_dir, final_exe)
dest_zip = "%s/%s" % (dest_dir, final_zip)

print("Building %s" % build_app)
args = ['pyinstaller', '-y', '--debug', '--windowed']
if onefile:
    args.append('--onefile')
args = ['pyinstaller', '-y']
args.append('%s.spec' % build_target)
run(args)

try:
    os.mkdir(dest_dir)
    print("Creating %s" % dest_dir)
except OSError:
    # Directory exists; remove old stuff
    if os.path.exists(dest_app):
        print("Removing old %s" % dest_app)
        shutil.rmtree(dest_app)
    if os.path.exists(dest_zip):
        print("Removing old %s" % dest_zip)
        os.unlink(dest_zip)
    if os.path.exists(dest_app):
        print("Removing old %s" % dest_exe)
        shutil.rmtree(dest_exe)

if win:
    if onefile:
        print("Copying %s -> %s" % (build_app, dest_exe))
        shutil.copyfile(build_app, dest_exe)
    else:
        print("One-folder build at: dist/%s" % build_target)
elif mac:
    contents = "%s/Contents" % build_app
    print("Copying new Info.plist")
    shutil.copyfile("Info.plist", "%s/Info.plist" % contents)

    dup = "%s/MacOS/libwx_osx_cocoau-3.0.dylib" % contents    
    if os.path.exists(dup):
        print("Fixing duplicate wxPython libs")
        os.unlink(dup)
        os.symlink("libwx_osx_cocoau-3.0.0.2.0.dylib", dup)

    print("Fixing missing symlink to geos library")
    os.symlink("libgeos_c.1.dylib", "%s/MacOS/libgeos_c.dylib" % contents)

    print("Copying %s -> %s & removing architectures other than x86_64" % (build_app, dest_app))
    #shutil.copytree(build_app, dest_app, True)
    run(['/usr/bin/ditto', '-arch', 'x86_64', build_app, dest_app])

    # print("Signing (with self-signed cert)")
    # run(["codesign", "-s", "test1", "--deep", dest_app])

    print("Signing NOAA Cert")
    run(["codesign",
         "-s",
         "Developer ID Application: National Oceanic and Atmospheric Administration",
         "--deep",
         "--force",
         "--timestamp=none",
         "--verbose",
         dest_app])

    print("Zipping %s" % dest_zip)
    run(['tar', 'cfj', dest_zip, '-C', dest_dir, final_app])

    print("Signing zip file")
    run(["codesign", "-s", "Developer ID Application: National Oceanic and Atmospheric Administration", "--deep", dest_zip])

# useful signing commands:
# spctl -a -t exec -vv MapRoom.app
# spctl -a -t open -vv dest_zip
