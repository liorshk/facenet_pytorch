import numpy as np
import shutil
import os
reps_path = '/media/lior/LinuxHDD/datasets/msceleb_rep.txt'

from_path = '/media/lior/LinuxHDD/datasets/MsCeleb-aligned'
to_path = '/media/lior/LinuxHDD/datasets/MSCeleb-cleaned'

reps = {}
with open(reps_path) as f:
    for line in f:
        path = line.split('+')[0]
        vector = np.array([float(x) for x in line.split('+')[1].strip().split(',')])
        folder = path.split('/')[0]
        file = path.split('/')[1]
        if folder not in reps:
            reps[folder] = {}

        reps[folder][file]=vector

def get_value_by_index(d,ix):
     return next( v for i, v in enumerate(d.items()) if i == ix )

for _dir in reps:
    X = []
    print(_dir)
    saved = None
    for _file in reps[_dir]:
        X.append(reps[_dir][_file])

    # Using Mean + STD
    i = 0
    mean = np.array(X).mean(axis=0)

    diff = np.array(X) - mean
    res = []
    for d in diff:
        res.append(np.dot(d,d))
    avg_dist = np.array(res).mean()
    std_dist = np.array(res).std()
    print("Average Distance {}, Std: {}".format(avg_dist,std_dist))
    if avg_dist > 0.5:
        print("BAD DIR: {}".format(_dir))
        continue

    for d in diff:
        if np.dot(d,d) > avg_dist+std_dist*2:
            print("BAD IMAGE: {}".format(get_value_by_index(reps[_dir],i)[0]))
        else:
            img_name = get_value_by_index(reps[_dir],i)[0]

            os.makedirs(os.path.join(to_path,_dir), exist_ok=True)
            shutil.copy(os.path.join(from_path,_dir,img_name),os.path.join(to_path,_dir,img_name))
        i+=1