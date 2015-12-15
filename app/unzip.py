import sys
import tarfile
import tempfile

def main():
    tarball_name = sys.argv[1]
    extract_dir = "./static/images/"
    tf = tarfile.open(name=tarball_name)
    tf.extractall(path=extract_dir)
    fileInfos = tf.getmembers()





if __name__ == '__main__':
    main()

