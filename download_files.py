"""
downloads files to the gallery
JJGC, February, 2017.
"""

import argparse
import sys
import os
from os import listdir
from os.path import isfile, join
import subprocess as sp
import datetime


def download_files(file_list):
    """download files in file_list the gallery"""

    path_to_gallery = '/data/icgallery/private/data/'
    server_name = 'icuser@next.ific.uv.es'
    path_to_ic = os.environ['IC_DATA']
    for ffile in file_list:
        print("""command:
                 scp {}:{}/{} {}
             """.format(server_name, path_to_gallery, ffile, path_to_ic))
        path_in = server_name + ':' + path_to_gallery + ffile
        sp.run(["scp", path_in, path_to_ic])

if __name__ == "__main__":

    if os.environ['IC_DATA'] == '':
        print("""Environment variable IC_DATA not defined.
                 Please define your IC_DATA pointing to the local
                 directory where you will store data and html files.""")
        sys.exit(1)


    parser = argparse.ArgumentParser(description='gallery parser.')
    parser.add_argument("file_list", nargs='+',
                        metavar="list of files", type=str,
                        help="list of files to be uploaded")


    args = parser.parse_args()
    #print(args)
    print(args.file_list)
    #for ffile in args.file_list:
    #    print(ffile)

    download_files(args.file_list)
