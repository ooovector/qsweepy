import os
import shutil

def spawn_plotting_script(dest,name):
	module_dir = os.path.dirname(os.path.abspath(__file__))
	shutil.copyfile(module_dir+"\\plotting_scripts\\"+name+'.py', dest+"\\"+name+'.py')
	shutil.copyfile(module_dir+"\\plotting_scripts\\plot.bat", dest+"\\plot.bat")
	bat = open(dest+"\\plot.bat", 'a')
	bat.write(' '+name+'.py')
	bat.close()