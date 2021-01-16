import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import zipfile
import wget
import shutil

if __name__ == '__main__':

    models_dirs = [
        'dante_by_char',
        'dante_by_syl',
        'dante_by_word',
        'dante_by_rev_syl',
        'dante_by_tonedrev_syl',
    ]

    working_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(working_dir, 'pretrained_models')

    for m in models_dirs:
        zip_name = '{}.zip'.format(m)
        data_zip = os.path.join(working_dir, zip_name)

        if not os.path.exists(data_zip):

            owner = 'luca-ant'
            repo = 'deep_comedy'

            url = 'https://github.com/{owner}/{repo}/releases/download/pretrained_models/{zip_name}'.format(owner=owner, repo=repo, zip_name=zip_name)
            print("DOWNLOADING {} PRETRAINED MODELS... ".format(m))
            try:
                zip_file = wget.download(url, os.path.join(working_dir, data_zip))
                print("\nDONE!")

            except:
                print("ERROR!")



        if os.path.exists(data_zip):

            print("EXTRACTING {}...".format(zip_name), end='\r')

            with zipfile.ZipFile(data_zip, 'r') as zip:
                zip.extractall()
            print("EXTRACTING {}... DONE!".format(zip_name))


            if os.path.exists(os.path.join(data_dir, m)):
                print("MOVING {} DATA TO DESTINATION FOLDER...".format(m), end='\r')

                log_dir_src = os.path.join(data_dir, m, 'logs')
                models_dir_src = os.path.join(data_dir, m, 'models')

                log_dir_dest = os.path.join(working_dir, m, 'logs')
                models_dir_dest = os.path.join(working_dir, m, 'models')
                shutil.rmtree(log_dir_dest, ignore_errors=True)
                shutil.rmtree(models_dir_dest, ignore_errors=True)

                if os.path.exists(log_dir_src):
                    shutil.move(log_dir_src, log_dir_dest)
                if os.path.exists(models_dir_src):
                    shutil.move(models_dir_src, models_dir_dest)
                print("MOVING {} DATA TO DESTINATION FOLDER... DONE!".format(m))

            shutil.rmtree(data_dir, ignore_errors=True)

        print()