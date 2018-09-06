import os
import sys
import shutil
import webbrowser
import multiprocessing as mp
import time
from pathlib import Path
from src.data import make_dataset
from src.features import build_features
from src.features import traintestsplit
from src.models import train_model
from src.visualization import visualize

def folder_structure(project_dir,root_dir):

    data_dir = os.path.join(project_dir,'data')
    if os.path.exists(root_dir):
        for name in os.listdir(root_dir):
            if name == "templates":
                src_dir = os.path.join(root_dir,name)
                dest_dir = os.path.join(project_dir,'src','models')
                shutil.move(src_dir,dest_dir)
            if name == ".env":
                dest_env_path = os.path.join(project_dir,name)
                src_env_path = os.path.join(root_dir,".env")
                shutil.move(src_env_path,dest_env_path)
    if not os.path.exists(data_dir):
        external_dir = os.path.join(data_dir,'external')
        os.makedirs(external_dir)
        interim_dir = os.path.join(data_dir,'interim')
        os.makedirs(interim_dir)
        processed_dir = os.path.join(data_dir,'processed')
        os.makedirs(processed_dir)
        raw_dir = os.path.join(data_dir,'raw')
        os.makedirs(raw_dir)

    models_dir = os.path.join(project_dir,'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    reports_dir = os.path.join(project_dir,'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    reports_dir = os.path.join(project_dir,'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        figures_dir = os.path.join(reports_dir,'figures')
        os.makedirs(figures_dir)

    templates_dir = os.path.join(project_dir,'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        
def exec_bat_file(project_dir):
    os.chdir(os.path.join(project_dir,'src','models'))
    os.system("python predict_model.py")

def launch_browser():
    url = 'http://127.0.0.1:5000/'
    webbrowser.open_new(url)

def main():
    project_dir = Path(__file__).resolve().parents[1]
    root_dir = sys.prefix
    folder_structure(project_dir,root_dir)
    make_dataset.main()
    build_features.main()
    traintestsplit.main()
    train_model.main()
    visualize.main()
    tasks = mp.cpu_count()
    pool  = mp.Pool(processes=tasks)
    pool.apply_async(exec_bat_file, (project_dir,), callback=None,error_callback=None)
    time.sleep(3)
    pool.apply_async(launch_browser, (), callback=None,error_callback=None)
    pool.close()
    pool.join()
    
