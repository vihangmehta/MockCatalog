import numpy as np

### HST Filters
hst_filters = {"wfc3": {"uvis2": ['f225w','f275w','f336w','f390w',
                                  'f438w','f475w','f555w','f606w',
                                  'f625w','f775w','f814w'],
                        "ir"   : ['f105w','f110w','f125w','f140w','f160w']},
               "acs" : {"wfc1" : ['f435w','f475w','f555w','f606w',
                                  'f625w','f775w','f814w']}}

### Luminosity Functions
LFs = [{"ref":"Scarlata+07","zmin":0.1,"zmax":0.6,"alpha":-1.26,"Mst":-21.03,"phi":-2.30,"wave":4420},
       {"ref":"Scarlata+07","zmin":0.6,"zmax":1.5,"alpha":-1.22,"Mst":-21.24,"phi":-2.31,"wave":4420},
       {"ref":"Mehta+17",   "zmin":1.5,"zmax":1.9,"alpha":-1.20,"Mst":-19.93,"phi":-2.12,"wave":1500},
       {"ref":"Mehta+17",   "zmin":1.9,"zmax":2.5,"alpha":-1.32,"Mst":-19.92,"phi":-2.30,"wave":1500},
       {"ref":"Mehta+17",   "zmin":2.5,"zmax":3.5,"alpha":-1.39,"Mst":-20.38,"phi":-2.42,"wave":1500},
       # {"ref":"Alavi+14",   "zmin":1.5,"zmax":2.7,"alpha":-1.74,"Mst":-20.01,"phi":-2.60,"wave":1500},
       # {"ref":"Reddy+09",   "zmin":2.7,"zmax":3.5,"alpha":-1.73,"Mst":-20.97,"phi":-2.77,"wave":1700},
       {"ref":"Bouwens+15", "zmin":3.5,"zmax":4.5,"alpha":-1.64,"Mst":-20.88,"phi":-2.71,"wave":1600},
       {"ref":"Bouwens+15", "zmin":4.5,"zmax":5.5,"alpha":-1.76,"Mst":-21.17,"phi":-3.13,"wave":1600}]
LFs = np.array(LFs)

### Survey Parameters ###
survey_area = 60 * 7.3 / 3600. # in sq.deg.
survey_zmin = 0.1
survey_zmax = 5.000001
survey_mlim = 27.0 # Magnitude limit at 1500A
