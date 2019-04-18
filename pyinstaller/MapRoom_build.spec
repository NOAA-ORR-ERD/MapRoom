# -*- mode: python -*-

block_cipher = None
DEBUG = False

# to pass -v to the python interpreter, uncomment this:
VERBOSE_INTERPRETER = False

if VERBOSE_INTERPRETER:
    options = [ ('v', None, 'OPTION'),]
else:
    options = []

appname = "MapRoom_build"

with open("../maproom.py", "r") as fh:
    script = fh.read()
with open("%s.py" % appname, "w") as fh:
    fh.write(script)

import sys
sys.modules['FixTk'] = None

import os
import pkg_resources

# On Windows 10, it can't find the ucrt DLLs without adding it to the search
# path. Doesn't affect other systems because this path won't exist.
import sys
sys.path.append("C:/Program Files (x86)/Windows Kits/10/Redist/ucrt/DLLs/x64")

pathex = [os.path.abspath("..")]

# Found sample code to include entry points inside the bundle at:
# https://github.com/pyinstaller/pyinstaller/issues/3050

hook_ep_packages = dict()
hiddenimports = set()
# List of packages that should have their Distutils entrypoints included.
ep_packages = ["sawx.loaders", "sawx.documents", "sawx.remember", "sawx.editors"]

if ep_packages:
    for ep_package in ep_packages:
        for ep in pkg_resources.iter_entry_points(ep_package):
            if ep_package in hook_ep_packages:
                package_entry_point = hook_ep_packages[ep_package]
            else:
                package_entry_point = []
                hook_ep_packages[ep_package] = package_entry_point
            package_entry_point.append(f"{ep.name} = {ep.module_name}")
            hiddenimports.add(ep.module_name)

    try:
        os.mkdir('./generated')
    except FileExistsError:
        pass

    for group, package_entry_point in hook_ep_packages.items():
        package_entry_point.sort()

    with open("./generated/pkg_resources_hook.py", "w") as f:
        f.write(f"""# Runtime hook generated from spec file to support pkg_resources entrypoints.
ep_packages = {hook_ep_packages}

if ep_packages:
    import pkg_resources
    default_iter_entry_points = pkg_resources.iter_entry_points

    def hook_iter_entry_points(group, name=None):
        if group in ep_packages and ep_packages[group]:
            eps = ep_packages[group]
            for ep in eps:
                parsedEp = pkg_resources.EntryPoint.parse(ep)
                parsedEp.dist = pkg_resources.Distribution()
                yield parsedEp
        else:
            return default_iter_entry_points(group, name)

    pkg_resources.iter_entry_points = hook_iter_entry_points
""")

print(hook_ep_packages)


a = Analysis(["%s.py" % appname],
             pathex=pathex,
             binaries=None,
             datas=None,
             hiddenimports=list(hiddenimports),
             hookspath=['.'],
             runtime_hooks=["./generated/pkg_resources_hook.py"],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'Cython', 'sphinx', 'nose', 'pygments', 'pytest'],
             cipher=block_cipher)

for pymod, path, tag in sorted(a.pure):
    if ".qt" in pymod or ".test" in pymod:
        print("why is this still here?", pymod)

# pytz zip bundle from https://github.com/pyinstaller/pyinstaller/wiki/Recipe-pytz-zip-file
# DOESN'T WORK ON MAC!

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

if sys.platform == "darwin":
    icon = '../resources/maproom.icns'
    exe = EXE(pyz,
        a.scripts,
        options,
        exclude_binaries=True,
        name=appname,
        debug=DEBUG,
        strip=not DEBUG,
        upx=not DEBUG,
        console=not DEBUG,
        icon=icon)
    coll = COLLECT(exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=not DEBUG,
        upx=not DEBUG,
        name=appname)
    app = BUNDLE(coll,
       name="%s.app" % appname,
       bundle_identifier="gov.noaa.maproom",
       icon=icon)

elif sys.platform == "win32":
    if not DEBUG:
        exe = EXE(pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,
            name=appname,
            debug=False,
            strip=False,
            upx=True,
            console=False,
            icon="../maproom/icons/maproom.ico")
    else:
        exe = EXE(pyz,
            a.scripts,
            options,
            exclude_binaries=True,
            name=appname,
            debug=True,
            strip=False,
            upx=True,
            console=True,
            icon="../maproom/icons/maproom.ico")
        coll = COLLECT(exe,
            a.binaries,
            a.zipfiles,
            a.datas,
            strip=False,
            upx=True,
            name=appname)

