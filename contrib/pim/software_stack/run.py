import os

os.system("sudo python3 setup.py install")
os.system("cp build/lib.linux-x86_64-3.6/* .")
os.system("python3 app.py")