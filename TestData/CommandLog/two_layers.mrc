# NOAA MapRoom Command File, v1
load TestData/Verdat/000689pts.verdat application/x-maproom-verdat
pt 1 -117.140778412 32.7241394552
pt 1 -117.100983652 32.7027306387
line 1 89 690
line_to 1 89 -117.122328114 32.71037724
line 1 689 691
move_pt 1 [689,691] 0.0115766575781 0.00550426670408
pt 1 -117.136437166 32.7112947877
load TestData/Verdat/000026pts.verdat application/x-maproom-verdat
pt 2 -48.930872705 41.3225828539
pt 2 -39.0488347613 41.8567326545
rename_layer 2 "26 points"
line 2 8 27
del 2 [] [16]
del 2 [] [13]
line 2 14 27
line 1 692 689
del 2 [27] []
del 2 [26] []
