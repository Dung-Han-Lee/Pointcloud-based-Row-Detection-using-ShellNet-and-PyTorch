from glob import glob
import config
import errno
import os

def unique_id(msg):
    ext = '.txt'
    f = glob("*"+ext)[0]
    num_trail = int(f.split(".")[0])
    newf = "./" + str(num_trail+1) + ext
    os.rename(f, newf)
    outdir = os.path.join("../weights", config.summary_prefix+"%02d"%num_trail)
    mkdir_p(outdir)
    f= open(outdir + "/msg.txt","w+")
    f.write(msg)
    f.close()
    return num_trail

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
