rem pyinstaller --additional-hooks-dir=pyinstaller/ pyinstaller/maproom.spec
pyinstaller pyinstaller/maproom.spec
copy pyinstaller\Microsoft.VC90.CRT-9.0.30729.6161\* dist\maproom
