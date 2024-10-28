from setuptools import find_packages,setup
from typing import List

# Indicator for requirements.txt
HYPHEN_E_DOT = "-e ."

# This function for return list of requirements
def get_requirements(file_path:str) -> List[str]:
    
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        # For removing '/n' and add empty
        requirements = [line.replace('/n',' ') for line in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
        
    

setup(
    
    name = 'cattle_management',
    version = '1.0.0',
    author = "Ahamed Basith",
    author_email = "alahamedbasithce@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)